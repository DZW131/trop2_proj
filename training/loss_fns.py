# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from contextlib import nullcontext
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
        parent_structure_idx=0,
        child_structure_idx=1,
        structure_contrast_temperature=0.1,
        structure_contrast_proj_dim=128,
        structure_contrast_hidden_dim=256,
        containment_eps=1e-6,
        ring_region_eps=1e-6,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0
        if "loss_struct_contrast" not in self.weight_dict:
            self.weight_dict["loss_struct_contrast"] = 0.0
        if "loss_contain" not in self.weight_dict:
            self.weight_dict["loss_contain"] = 0.0
        if "loss_ring" not in self.weight_dict:
            self.weight_dict["loss_ring"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.parent_structure_idx = parent_structure_idx
        self.child_structure_idx = child_structure_idx
        self.structure_contrast_temperature = structure_contrast_temperature
        self.containment_eps = containment_eps
        self.ring_region_eps = ring_region_eps
        if self.weight_dict["loss_struct_contrast"] > 0:
            self.structure_projection = nn.Sequential(
                nn.LazyLinear(structure_contrast_hidden_dim),
                nn.GELU(),
                nn.Linear(structure_contrast_hidden_dim, structure_contrast_proj_dim),
            )
        else:
            self.structure_projection = nn.Identity()

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """

        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 5  # [N, 1, task_num, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {
            "loss_mask": 0,
            "loss_dice": 0,
            "loss_iou": 0,
            "loss_class": 0,
            "loss_struct_contrast": self._zero_loss(targets),
            "loss_contain": self._zero_loss(targets),
            "loss_ring": self._zero_loss(targets),
        }
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )
        losses["loss_struct_contrast"] = self._structure_contrastive_loss(
            outputs, targets
        )
        losses["loss_contain"] = self._containment_loss(outputs, targets)
        losses["loss_ring"] = self._ring_region_consistency_loss(outputs, targets)
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _zero_loss(self, targets: torch.Tensor) -> torch.Tensor:
        return targets.new_zeros((), dtype=torch.float32)

    def _valid_structure_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if (
            targets.dim() != 4
            or targets.size(1) <= self.parent_structure_idx
            or targets.size(1) <= self.child_structure_idx
        ):
            return torch.zeros(targets.size(0), dtype=torch.bool, device=targets.device)
        parent_present = torch.any(
            (targets[:, self.parent_structure_idx] > 0).flatten(1), dim=-1
        )
        child_present = torch.any(
            (targets[:, self.child_structure_idx] > 0).flatten(1), dim=-1
        )
        return parent_present & child_present

    def _structure_contrastive_loss(
        self, outputs: Dict, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.weight_dict["loss_struct_contrast"] <= 0:
            return self._zero_loss(targets)
        obj_ptr = outputs.get("obj_ptr", None)
        if obj_ptr is None:
            return self._zero_loss(targets)
        if obj_ptr.dim() == 2:
            num_structures = targets.size(1)
            if num_structures <= 0 or obj_ptr.size(1) % num_structures != 0:
                return self._zero_loss(targets)
            obj_ptr = obj_ptr.view(obj_ptr.size(0), num_structures, -1)
        if obj_ptr.dim() != 3:
            return self._zero_loss(targets)
        if (
            obj_ptr.size(1) <= self.parent_structure_idx
            or obj_ptr.size(1) <= self.child_structure_idx
        ):
            return self._zero_loss(targets)

        valid = self._valid_structure_targets(targets)
        if valid.sum().item() < 2:
            return self._zero_loss(targets)

        parent_ptr = obj_ptr[valid, self.parent_structure_idx].float()
        child_ptr = obj_ptr[valid, self.child_structure_idx].float()
        parent_ptr = self.structure_projection(parent_ptr)
        child_ptr = self.structure_projection(child_ptr)
        parent_ptr = F.normalize(parent_ptr, dim=-1)
        child_ptr = F.normalize(child_ptr, dim=-1)

        logits = torch.matmul(parent_ptr, child_ptr.transpose(0, 1))
        logits = logits / self.structure_contrast_temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_parent = F.cross_entropy(logits, labels)
        loss_child = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_parent + loss_child)

    def _containment_loss(self, outputs: Dict, targets: torch.Tensor) -> torch.Tensor:
        if self.weight_dict["loss_contain"] <= 0:
            return self._zero_loss(targets)
        pred_masks = outputs.get("pred_masks_high_res", None)
        if pred_masks is None or pred_masks.dim() != 4:
            return self._zero_loss(targets)
        if (
            pred_masks.size(1) <= self.parent_structure_idx
            or pred_masks.size(1) <= self.child_structure_idx
        ):
            return self._zero_loss(targets)

        valid = self._valid_structure_targets(targets)
        if not torch.any(valid).item():
            return self._zero_loss(targets)

        parent_prob = pred_masks[:, self.parent_structure_idx].float().sigmoid()
        child_prob = pred_masks[:, self.child_structure_idx].float().sigmoid()
        outside_mass = (child_prob * (1 - parent_prob)).flatten(1).sum(-1)
        child_mass = child_prob.flatten(1).sum(-1)
        contain_error = outside_mass / (child_mass + self.containment_eps)
        return contain_error[valid].mean()

    def _ring_region_consistency_loss(
        self, outputs: Dict, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.weight_dict["loss_ring"] <= 0:
            return self._zero_loss(targets)
        pred_masks = outputs.get("pred_masks_high_res", None)
        if pred_masks is None or pred_masks.dim() != 4:
            return self._zero_loss(targets)
        if (
            pred_masks.size(1) <= self.parent_structure_idx
            or pred_masks.size(1) <= self.child_structure_idx
        ):
            return self._zero_loss(targets)

        valid = self._valid_structure_targets(targets)
        if not torch.any(valid).item():
            return self._zero_loss(targets)

        parent_prob = pred_masks[:, self.parent_structure_idx].float().sigmoid()
        child_prob = pred_masks[:, self.child_structure_idx].float().sigmoid()
        ring_prob = parent_prob * (1 - child_prob)

        parent_target = (targets[:, self.parent_structure_idx] > 0).float()
        child_target = (targets[:, self.child_structure_idx] > 0).float()
        ring_target = torch.clamp(parent_target - child_target, min=0.0, max=1.0)

        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=False)
            if pred_masks.device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            ring_prob_fp32 = ring_prob.float().clamp(
                min=self.ring_region_eps, max=1.0 - self.ring_region_eps
            )
            ring_target_fp32 = ring_target.float()
            bce = F.binary_cross_entropy(
                ring_prob_fp32, ring_target_fp32, reduction="none"
            )
            bce = bce.flatten(1).mean(-1)

        intersection = (ring_prob_fp32 * ring_target_fp32).flatten(1).sum(-1)
        denominator = (
            ring_prob_fp32.flatten(1).sum(-1) + ring_target_fp32.flatten(1).sum(-1)
        )
        dice = 1 - (2 * intersection + 1.0) / (denominator + 1.0 + self.ring_region_eps)

        return (bce + dice)[valid].mean()

    def _dilate_masks(self, masks, dilation_size=5):
        """
        Dilate the masks by a given dilation size.
        """
        kernel = torch.ones(
            (1, 1, dilation_size, dilation_size),
            dtype=masks.dtype,
            device=masks.device,
        )
        return F.conv2d(masks, kernel, padding=dilation_size // 2).bool().to(torch.float32)

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        # target_masks = target_masks.expand_as(src_masks)
        # dilate_gt = self._dilate_masks(target_masks)
        # dilate_gt = self._dilate_masks(dilate_gt)
        # target_masks = torch.cat((target_masks, dilate_gt), dim=1)
        target_masks = target_masks.squeeze(1)
        if target_masks.shape!=src_masks.shape:
            target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks > 0).flatten(2), dim=-1).float()
            object_score_logits = object_score_logits.expand_as(target_obj)
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        if loss_multimask.size(1) > 2:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
