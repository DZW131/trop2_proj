import argparse
import csv
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from infer_utils import *
from omegaconf import OmegaConf
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.amg import (
    MaskData,
    batched_mask_to_box,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    uncrop_masks,
)
from tqdm import tqdm
from training.utils.experiment_utils import (
    ABLATION_PRESETS,
    DEFAULT_INFER_CONFIG,
    default_checkpoint_path,
    infer_ablation_name,
)
from training.utils.utils import count_trainable_params
from utils import *

MEMBRANE_LABEL = "\u80bf\u7624\u7ec6\u80de\u819c"
NUCLEUS_LABEL = "\u80bf\u7624\u7ec6\u80de\u6838"

LEGACY_MODELS = {
    "custom": {
        "ckpt_path": "checkpoints/mul_query/checkpoints/checkpoint.pt",
        "config_file": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "label": [MEMBRANE_LABEL, NUCLEUS_LABEL],
        "order": [0, 1],
        "multitask_num": 2,
    },
    "large": {
        "ckpt_path": "checkpoints/sam2.1_hiera_large.pt",
        "config_file": "configs/sam2.1/sam2.1_hiera_l.yaml",
    },
    "bplus_me": {
        "ckpt_path": "checkpoints/1118_me/checkpoints/checkpoint.pt",
        "config_file": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "label": [MEMBRANE_LABEL],
        "order": [0],
        "multitask_num": 1,
    },
    "bplus_nu": {
        "ckpt_path": "checkpoints/1118_nu/checkpoints/checkpoint.pt",
        "config_file": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "label": [NUCLEUS_LABEL],
        "order": [0],
        "multitask_num": 1,
    },
    "bplus_menu": {
        "ckpt_path": "checkpoints/1118_me_nu2/checkpoints/checkpoint.pt",
        "config_file": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "label": [MEMBRANE_LABEL, NUCLEUS_LABEL],
        "order": [0, 1],
        "multitask_num": 2,
    },
}

cfg = OmegaConf.create(LEGACY_MODELS)
ACTIVE_MODEL_CFG = None
ACTIVE_MODEL_TAG = None
predictor = None
PER_CASE_FIELDS = [
    "experiment",
    "split",
    "sample_id",
    "image_path",
    "json_path",
    "label",
    "bdq",
    "bsq",
    "bpq",
    "aji",
]


def require_labeled_model_cfg():
    if ACTIVE_MODEL_CFG is None:
        raise RuntimeError("Model is not loaded yet.")
    if "label" not in ACTIVE_MODEL_CFG or "order" not in ACTIVE_MODEL_CFG:
        raise ValueError(
            "The selected model config does not define dataset labels or decoder order. "
            "Use --model bplus_me, --model bplus_nu, --model bplus_menu, or provide "
            "--ablation with a paired checkpoint."
        )
    return ACTIVE_MODEL_CFG


def resolve_model_entry(args):
    global ACTIVE_MODEL_TAG

    resolved_ablation = infer_ablation_name(
        ablation=args.ablation,
        with_contain=args.with_contain,
        with_contrast=args.with_contrast,
    )

    if resolved_ablation is not None:
        model_entry = OmegaConf.to_container(cfg["bplus_menu"], resolve=True)
        model_entry["ckpt_path"] = args.ckpt_path or default_checkpoint_path(
            resolved_ablation
        )
        model_entry["config_file"] = args.config_file or DEFAULT_INFER_CONFIG
        ACTIVE_MODEL_TAG = args.experiment_tag or resolved_ablation
        return OmegaConf.create(model_entry)

    model_key = args.model
    model_entry = OmegaConf.to_container(cfg[model_key], resolve=True)
    if args.ckpt_path is not None:
        model_entry["ckpt_path"] = args.ckpt_path
    if args.config_file is not None:
        model_entry["config_file"] = args.config_file
    ACTIVE_MODEL_TAG = args.experiment_tag or model_key
    return OmegaConf.create(model_entry)


def load_model(args):
    global predictor
    global ACTIVE_MODEL_CFG

    ACTIVE_MODEL_CFG = resolve_model_entry(args)
    if not os.path.exists(ACTIVE_MODEL_CFG.ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ACTIVE_MODEL_CFG.ckpt_path}. "
            "Use --ckpt-path to override or train the matching ablation first."
        )
    predictor = SAM2ImagePredictor(build_sam2(**ACTIVE_MODEL_CFG))
    count_trainable_params(predictor.model)


def resolve_prompt_label(args):
    if args.prompt_label is not None:
        return args.prompt_label
    model_cfg = require_labeled_model_cfg()
    return model_cfg.label[0]


def sanitize_tag(value):
    return str(value).replace("/", "_").replace("\\", "_")


def collect_shapes_by_label(shapes, label):
    return [item["points"] for item in shapes if item.get("label") == label]


def build_output_payload(template, predicted_labels):
    predicted_labels = set(predicted_labels)
    preserved_shapes = [
        item for item in template.get("shapes", []) if item.get("label") not in predicted_labels
    ]
    payload = {key: value for key, value in template.items() if key != "shapes"}
    payload["shapes"] = preserved_shapes
    return payload


def save_eval_artifacts(args, per_case_rows, summary):
    if not args.save_metrics:
        return

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    experiment_tag = sanitize_tag(ACTIVE_MODEL_TAG)
    stem = f"{experiment_tag}_{args.mode}"

    csv_path = metrics_dir / f"{stem}_per_case.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PER_CASE_FIELDS)
        writer.writeheader()
        writer.writerows(per_case_rows)
    with open(metrics_dir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def inference(prompts, key, image_hw, order=0, step=1, multimask_output=False):
    assert key in prompts, f"{key} not in prompts"
    all_masks, all_scores = [], []
    all_key_prompt = np.concatenate(prompts[key], axis=0)
    mask_data = MaskData()
    orig_h, orig_w = image_hw

    for prompt_id in np.unique(all_key_prompt[:, 2]):
        point_coords = all_key_prompt[all_key_prompt[:, 2] == prompt_id][:, :2][:, None, :]
        point_labels = np.ones(point_coords.shape[0])[:, None]
        mask_threshold = 0.0
        start_time = time.time()
        masks1, scores1, _ = predictor.predict(
            point_coords, point_labels, multimask_output=multimask_output
        )
        print(f"time: {time.time() - start_time}")
        if multimask_output:
            masks1 = [masks1[0][:, 0:1, :, :], masks1[0][:, 1:2, :, :]]
            scores1 = [scores1[0][:, 0:1], scores1[0][:, 1:2]]
        if len(masks1) > 1:
            contained = 0
            for membrane_mask, nucleus_mask in zip(masks1[0], masks1[1]):
                if assert_mask_contained(membrane_mask[0], nucleus_mask[0]):
                    contained += 1
            print(f"containment ratio: {contained / len(masks1[0]) * 100:.2f}%")

        for label_index, (masks, scores) in enumerate(
            zip(masks1[order : order + step], scores1[order : order + step])
        ):
            batch_data = MaskData(
                masks=(
                    torch.from_numpy(masks).squeeze(1)
                    if masks.ndim == 4
                    else torch.from_numpy(masks)
                ),
                iou_preds=(
                    torch.from_numpy(scores).squeeze(1)
                    if masks.ndim == 4
                    else torch.from_numpy(scores)
                ),
                points=(
                    torch.from_numpy(point_coords)
                    if masks.ndim == 4
                    else torch.from_numpy(point_coords).squeeze(1)
                ),
                categories=torch.tensor([0] * masks.shape[0]),
                inds=torch.arange(masks.shape[0]),
                labels=torch.tensor([label_index] * masks.shape[0]),
            )
            batch_data["masks"] = batch_data["masks"] > mask_threshold
            batch_data["boxes"] = batched_mask_to_box(batch_data["masks"])
            keep_mask = ~is_box_near_crop_edge(
                batch_data["boxes"],
                [0, 0, orig_w, orig_h],
                [0, 0, orig_w, orig_h],
                atol=7,
            )
            if not torch.all(keep_mask):
                batch_data.filter(keep_mask)
            batch_data["masks"] = uncrop_masks(
                batch_data["masks"], [0, 0, orig_w, orig_h], orig_h, orig_w
            )
            batch_data["rles"] = mask_to_rle_pytorch(batch_data["masks"])
            all_masks.append(masks)
            all_scores.append(scores)
            mask_data.cat(batch_data)

    masks = np.concatenate(all_masks, axis=0)
    scores = np.concatenate(all_scores, axis=0).reshape(-1)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    return mask_data, masks, scores


def evaluate(img_path, json_path, keys=None, save_res=False):
    model_cfg = require_labeled_model_cfg()
    keys = keys or [model_cfg.label[0]]
    target_label = keys[0]

    img = Image.open(img_path)
    image = np.array(img.copy().convert("RGB"))
    with open(json_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    data = {target_label: collect_shapes_by_label(template["shapes"], target_label)}
    prompts = get_prompt(data, {}, target_label)
    print("prompt count:", sum(len(prompts[key]) for key in prompts))
    predictor.set_image(image)
    mask_data, _, _ = inference(
        prompts,
        target_label,
        image.shape[:2],
        order=model_cfg.order[model_cfg.label.index(target_label)],
        multimask_output=False,
    )
    inst_maps = get_inst_maps(image, data[target_label])
    mask_data = data_format(mask_data, crop_box=[0, 0, image.shape[1], image.shape[0]])
    bdq_tmp, bsq_tmp, bpq_tmp, aji_score = cal_metric(
        inst_maps, mask_data, image.shape[0], image.shape[1]
    )
    if save_res:
        vis_image = np.array(img.copy().convert("RGB"))
        for ann in mask_data:
            contours, _ = cv2.findContours(
                ann["segmentation"].astype(np.uint8),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(vis_image, contours, -1, rand_color(), -1)
        cv2.imwrite(f"{target_label}_1.jpg", vis_image[:, :, ::-1])
    return bdq_tmp, bsq_tmp, bpq_tmp, aji_score


def infer(img_path, save_res=True, prompt_json_path=None, prompt_label=None):
    model_cfg = require_labeled_model_cfg()
    prompt_json_path = prompt_json_path or str(Path(img_path).with_suffix(".json"))
    prompt_label = prompt_label or model_cfg.label[0]

    img = Image.open(img_path)
    image = np.array(img.copy().convert("RGB"))

    with open(prompt_json_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    prompt_shapes = collect_shapes_by_label(template["shapes"], prompt_label)
    if not prompt_shapes:
        raise ValueError(
            f"No prompt shapes found for label '{prompt_label}' in {prompt_json_path}."
        )
    prompts = get_prompt({"key": prompt_shapes}, {}, "key")

    start = time.time()
    predictor.set_image(image)
    mask_data, _, _ = inference(
        prompts, "key", image.shape[:2], step=len(model_cfg.label)
    )
    elapsed_ms = (time.time() - start) * 1000
    print(f"time: {elapsed_ms} ms")

    mask_data["output"] = build_output_payload(template, model_cfg.label)

    if save_res:
        model_suffix = sanitize_tag(ACTIVE_MODEL_TAG)
        for label_index, label_name in enumerate(model_cfg.label):
            vis_image = np.array(img.copy().convert("RGB"))
            for mask_index, (mask, pred_label_index) in enumerate(
                zip(mask_data["masks"].numpy(), mask_data["labels"].numpy().tolist())
            ):
                if mask_data["iou_preds"][mask_index] < 0.5:
                    continue
                if pred_label_index != label_index:
                    continue

                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                if len(contours) != 1 or contours[0].shape[0] < 3:
                    print(label_index, mask_index, len(contours))
                    continue

                cv2.drawContours(vis_image, contours, -1, rand_color(), -1)
                mask_data["output"]["shapes"].append(
                    {
                        "points": contours[0].reshape(-1, 2).tolist(),
                        "label": label_name,
                        "score": mask_data["iou_preds"][mask_index].item(),
                        "shape_type": "polygon",
                        "description": f"{mask_index}",
                        "mask": None,
                        "flags": {},
                        "group_id": None,
                    }
                )

            save_path = img_path.replace(".png", f"_{model_suffix}_{label_index}.png")
            cv2.imwrite(save_path, vis_image[:, :, ::-1])
            print(f"save res to {save_path}")

        json_save_path = img_path.replace(".png", f"_{model_suffix}.json")
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(mask_data["output"], f, indent=4, ensure_ascii=False)

    return mask_data


def main(args):
    model_cfg = require_labeled_model_cfg()
    mode = args.mode
    img_root = f"datasets/trop2/{mode}/JPEGImages"
    json_root = f"datasets/trop2/{mode}/jsons"
    img_list = glob.glob(os.path.join(img_root, "**/*.png"), recursive=True)
    json_list = glob.glob(os.path.join(json_root, "**/*.json"), recursive=True)
    img_list.sort()
    json_list.sort()

    metrics = {key: [] for key in model_cfg.label}
    per_case_rows = []
    for img_path, json_path in tqdm(zip(img_list, json_list), total=len(img_list)):
        img_case = Path(img_path).parent.name
        json_case = Path(json_path).parent.name
        if img_case != json_case:
            print(img_path, json_path)
            continue
        for key in metrics.keys():
            bdq_tmp, bsq_tmp, bpq_tmp, aji_score = evaluate(
                img_path, json_path, keys=[key]
            )
            metrics[key].append((bdq_tmp, bsq_tmp, bpq_tmp, aji_score))
            per_case_rows.append(
                {
                    "experiment": ACTIVE_MODEL_TAG,
                    "split": mode,
                    "sample_id": img_case,
                    "image_path": img_path,
                    "json_path": json_path,
                    "label": key,
                    "bdq": bdq_tmp,
                    "bsq": bsq_tmp,
                    "bpq": bpq_tmp,
                    "aji": aji_score,
                }
            )

    summary = {}
    metric_names = ("bdq", "bsq", "bpq", "aji")
    for key, values in metrics.items():
        if len(values) == 0:
            summary[key] = {metric_name: float("nan") for metric_name in metric_names}
            continue
        mean_values = np.nanmean(values, axis=0)
        summary[key] = {
            metric_name: float(metric_value)
            for metric_name, metric_value in zip(metric_names, mean_values)
        }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    save_eval_artifacts(args, per_case_rows, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--save_res", action="store_true")
    parser.add_argument("--img_path", type=str, default="assets/0000.png")
    parser.add_argument(
        "--prompt-json",
        type=str,
        default=None,
        help="optional prompt annotation json for single-image inference",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=sorted(ABLATION_PRESETS.keys()),
        default=None,
        help="select ablation profile and load its default checkpoint path",
    )
    parser.add_argument(
        "--with-contain",
        action="store_true",
        help="infer with the containment-only experiment profile",
    )
    parser.add_argument(
        "--with-contrast",
        action="store_true",
        help="infer with the contrast-only experiment profile",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="override checkpoint path; if --ablation is used, defaults to checkpoints/ablations/<ablation>/checkpoints/checkpoint.pt",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help=f"override inference model config file (default ablation config file: {DEFAULT_INFER_CONFIG})",
    )
    parser.add_argument(
        "--experiment-tag",
        type=str,
        default=None,
        help="tag used for saved outputs and metric files",
    )
    parser.add_argument(
        "--prompt-label",
        type=str,
        default=None,
        help="override the prompt label used to extract clicks from the input json",
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="save per-case CSV and summary JSON during --eval",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="analysis/eval",
        help="directory for exported evaluation metrics",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bplus_menu",
        help=", ".join(list(LEGACY_MODELS.keys())),
    )

    args = parser.parse_args()
    load_model(args)
    if args.eval:
        main(args)
    else:
        infer(
            args.img_path,
            save_res=args.save_res,
            prompt_json_path=args.prompt_json,
            prompt_label=resolve_prompt_label(args),
        )
