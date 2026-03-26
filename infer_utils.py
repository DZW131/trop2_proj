import cv2
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms
from utils import *
from sam2.utils.amg import (
    area_from_rle,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    MaskData,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
from shapely.geometry import Polygon


def get_point(item):
    try:
        M = cv2.moments(np.array(item))
        if M["m00"]:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
    except:
        cx, cy = item[0]
    return np.array([cx, cy])


def points_to_point(points):
    try:
        return get_point(points)
    except:
        return np.array(points)[:1]


def get_prompt(data, prompts, key):

    for i, item in enumerate(data[key]):
        slct_point = points_to_point(item)

        if len(slct_point.shape) == 1:
            slct_point = slct_point[None, :]
        assert (
            slct_point.shape[0]
        ) == 1, f"slct_point.shape[0] is {slct_point.shape[0]}"
        idx = np.ones(slct_point.shape[0]) * 1
        slct_point_with_id = np.concatenate([slct_point, idx[:, None]], axis=1)
        if key not in prompts:
            prompts[key] = []
        prompts[key].extend([slct_point_with_id])
    return prompts


def get_inst_maps(image, data):

    inst_maps = np.ascontiguousarray(np.zeros_like(image)[:, :, 0])
    for i, pt in enumerate(data):
        region = np.array(pt).astype(np.int32)
        if region.shape[0] > 5:
            cv2.fillPoly(inst_maps, [region], i + 1)
    return inst_maps


def cal_metric(inst_maps, curr_anns, orig_h, orig_w):
    all_masks = []
    all_boxes = []
    all_scores = []
    all_classes = []
    all_inds = []
    for mask_data in curr_anns:
        all_scores.append(mask_data["predicted_iou"])
        all_masks.append(mask_data["segmentation"][:orig_h, :orig_w])
        all_boxes.append(mask_data["bbox"])
        all_classes.append(mask_data["categories"])

        all_inds.append(mask_data["inds"])
    all_boxes = torch.as_tensor(all_boxes)
    all_scores = torch.as_tensor(all_scores)

    all_inds = np.asarray(all_inds)
    unique_inds, counts = np.unique(all_inds, return_counts=True)

    # first-aspect NMS
    keep_prior = np.ones(len(all_inds), dtype=bool)
    for i in np.where(counts > 1)[0]:
        inds = np.where(all_inds == unique_inds[i])[0]
        inds = np.delete(inds, np.argmax(all_scores[inds]))
        keep_prior[inds] = False
    keep_prior = torch.from_numpy(keep_prior)

    all_boxes = all_boxes[keep_prior]
    all_scores = all_scores[keep_prior]
    all_masks = [all_masks[ind] for ind in np.where(keep_prior)[0]]
    iou_threshold = 0.5
    # second-aspect NMS
    keep_by_nms = batched_nms(
        all_boxes.float(),
        all_scores,
        torch.zeros_like(all_boxes[:, 0]),  # apply cross categories
        iou_threshold=iou_threshold,
    ).numpy()
    order = keep_by_nms[::-1]
    b_inst_map = np.zeros_like(inst_maps, dtype=int)
    for iid, ind in enumerate(order):
        b_inst_map[all_masks[ind]] = iid + 1

    if len(np.unique(inst_maps)) == 1:
        bpq_tmp = np.nan
        bdq_tmp = np.nan
        bsq_tmp = np.nan
    else:
        [bdq_tmp, bsq_tmp, bpq_tmp], _ = get_fast_pq(
            remap_label(inst_maps), remap_label(b_inst_map)
        )
    aji_score = get_fast_aji(remap_label(inst_maps), remap_label(b_inst_map))

    return bdq_tmp, bsq_tmp, bpq_tmp, aji_score


def data_format(mask_data, box_nms_thresh=1.0, crop_box=None):
    # 数据类型转换
    keep_by_nms = batched_nms(
        mask_data["boxes"].float(),
        mask_data["iou_preds"],
        torch.zeros_like(mask_data["boxes"][:, 0]),  # apply cross categories
        iou_threshold=box_nms_thresh,
    )
    mask_data.filter(keep_by_nms)

    # Return to the original image frame
    mask_data["boxes"] = uncrop_boxes_xyxy(mask_data["boxes"], crop_box)
    mask_data["points"] = uncrop_points(mask_data["points"], crop_box)
    mask_data["crop_boxes"] = torch.tensor(
        [crop_box for _ in range(len(mask_data["rles"]))]
    )

    mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]

    # Write mask records
    curr_anns = []
    for idx in range(len(mask_data["segmentations"])):
        ann = {
            "segmentation": mask_data["segmentations"][idx],
            "area": area_from_rle(mask_data["rles"][idx]),
            "bbox": mask_data["boxes"][idx].tolist(),
            "predicted_iou": mask_data["iou_preds"][idx].item(),
            "point_coords": [mask_data["points"][idx].tolist()],
            "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            "categories": mask_data["categories"][idx].tolist(),
            "inds": mask_data["inds"][idx].tolist(),
        }
        curr_anns.append(ann)
    return curr_anns


def is_polygon_contained(polyA, polyB):
    # 判断多边形A是否完全包含在多边形B中
    A = Polygon(polyA)
    B = Polygon(polyB)
    return A.within(B) or A.equals(B)


def assert_mask_contained(maskA: np.ndarray, maskB: np.ndarray):

    # 确保maskA和maskB都是多边形
    # maskB 在 maskA 中
    assert maskA.ndim == 2 and maskB.ndim == 2, "maskA and maskB must be 2D arrays"
    union_mask = np.logical_and(maskA, maskB)
    return np.all(union_mask == maskB)
