import glob
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from infer_utils import *
from utils import *
from omegaconf import OmegaConf
from sam2.utils.amg import (
    batched_mask_to_box,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    MaskData,
    uncrop_masks,
)
from tqdm import tqdm
from training.utils.utils import count_trainable_params

config = {
    "custom": {
        "ckpt_path": "checkpoints/mul_query/checkpoints/checkpoint.pt",
        "config_file": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "label": ["肿瘤细胞膜", "肿瘤细胞核"],
        "order": [0, 1],
        "multitask_num": 1,
    },
    "large": {
        "ckpt_path": "checkpoints/sam2.1_hiera_large.pt",
        "config_file": "configs/sam2.1/sam2.1_hiera_l.yaml",
    },
    "bplus_me": {
        "ckpt_path": "checkpoints/1118_me/checkpoints/checkpoint.pt",
        "config_file": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "label": ["肿瘤细胞膜"],
        "order": [0],
        "multitask_num": 1,
    },
    "bplus_nu": {
        "ckpt_path": "checkpoints/1118_nu/checkpoints/checkpoint.pt",
        "config_file": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "label": ["肿瘤细胞核"],
        "order": [0],
        "multitask_num": 1,
    },
    "bplus_menu": {
        "ckpt_path": "checkpoints/1118_me_nu2/checkpoints/checkpoint.pt",
        "config_file": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "label": ["肿瘤细胞膜", "肿瘤细胞核"],
        "order": [0, 1],
        "multitask_num": 2,
    },
}
cfg = OmegaConf.create(config)


def load_model(args):
    global predictor
    predictor = SAM2ImagePredictor(build_sam2(**cfg[args.model]))
    # for name, m in predictor.model.sam_mask_decoders.named_modules():
    #     m.register_forward_hook(stat_hook(name))

    count_trainable_params(predictor.model)


def inference(prompts, key, wh, order=0, step=1, multimask_output=False):
    # for key in keys:
    assert key in prompts, f"{key} not in prompts"
    all_masks, all_scores = [], []
    all_key_prompt = np.concatenate(prompts[key], axis=0)
    mask_data = MaskData()
    for i in np.unique(all_key_prompt[:, 2]):
        point_coords = all_key_prompt[all_key_prompt[:, 2] == i][:, :2][:, None, :]
        point_labels = np.ones(point_coords.shape[0])[:, None]
        mask_threshold = 0.0
        start_time = time.time()
        # masks, scores, logits = predictor.predict(point_coords, point_labels, multimask_output=False)
        masks1, scores1, logits = predictor.predict(
            point_coords, point_labels, multimask_output=multimask_output
        )
        print(f"time: {time.time()-start_time}")
        if multimask_output:
            masks1 = [masks1[0][:, 0:1, :, :], masks1[0][:, 1:2, :, :]]
            scores1 = [scores1[0][:, 0:1], scores1[0][:, 1:2]]
        if len(masks1) > 1:
            # 判断包含关系，细胞膜一定要包含细胞核
            sums = 0
            for mask1, mask2 in zip(masks1[0], masks1[1]):
                if assert_mask_contained(mask1[0], mask2[0]):
                    sums += 1
            print(f"完全包含率：{sums/len(masks1[0])*100:.2f}%")
        for l, (masks, scores) in enumerate(
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
                labels=torch.tensor([l] * masks.shape[0]),
            )
            orig_w, orig_h = wh
            # Threshold masks and calculate boxes
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
            # Compress to RLE
            batch_data["masks"] = uncrop_masks(
                batch_data["masks"], [0, 0, orig_w, orig_h], orig_h, orig_w
            )
            batch_data["rles"] = mask_to_rle_pytorch(batch_data["masks"])
            all_masks.append(masks)
            all_scores.append(scores)
            mask_data.cat(batch_data)
    masks = np.concatenate(all_masks, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    return mask_data, masks, scores


def evaluate(img_path, json_path, keys=["肿瘤细胞膜"], save_res=False):
    img = Image.open(img_path)
    image = np.array(img.copy().convert("RGB"))
    # color_map = {key: rand_color() for key in keys}
    data = {
        keys[0]: [
            item["points"]
            for item in json.load(open(json_path))["shapes"]
            if item["label"] == keys[0]
        ]
    }

    prompts = {}
    prompts = get_prompt(data, prompts, keys[0])
    print("prompt 数量：", sum([len(prompts[key]) for key in prompts]))
    predictor.set_image(image)
    mask_data, _, _ = inference(
        prompts,
        keys[0],
        image.shape[:2],
        order=cfg[args.model].order[cfg[args.model].label.index(keys[0])],
        multimask_output=False,
    )
    inst_maps = get_inst_maps(image, data[keys[0]])
    mask_data = data_format(mask_data, crop_box=[0, 0, image.shape[1], image.shape[0]])
    bdq_tmp, bsq_tmp, bpq_tmp, aji_score = cal_metric(
        inst_maps, mask_data, image.shape[0], image.shape[1]
    )
    if save_res:
        image = np.array(img.copy().convert("RGB"))
        [
            cv2.drawContours(
                image,
                cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )[0],
                -1,
                rand_color(),
                -1,
            )
            for mask in mask_data["masks"].numpy()
        ]
        cv2.imwrite(f"{keys[0]}_1.jpg", image[:, :, ::-1])
    return bdq_tmp, bsq_tmp, bpq_tmp, aji_score


def infer(img_path, save_res=True):
    # 准备图像
    prompts_json_path = img_path.replace(".png", ".json")
    img = Image.open(img_path)
    image = np.array(img.copy().convert("RGB"))

    with open(prompts_json_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    data = {
        "key": [
            item["points"]
            for item in template["shapes"]
            if item["label"] == "肿瘤细胞膜"
        ]
    }
    prompts = {}
    prompts = get_prompt(data, prompts, "key")

    start = time.time()
    times = 1
    for _ in range(times):
        predictor.set_image(image)
        mask_data, _, _ = inference(
            prompts, "key", image.shape[:2], step=len(cfg[args.model].label)
        )
    end = time.time()
    print(f"time: {(end - start) / times * 1000} ms")

    shapes = template["shapes"]
    del template["shapes"]
    mask_data["output"] = {
        **template,
        "shapes": [item for item in shapes if item["label"] == "阳性肿瘤细胞"],
    }

    if save_res:
        model_suffix = args.model.split("_")[-1]

        for label in range(len(cfg[args.model].label)):
            image = np.array(img.copy().convert("RGB"))
            for i, (mask, label_ind) in enumerate(
                zip(mask_data["masks"].numpy(), mask_data["labels"].numpy().tolist())
            ):
                if mask_data["iou_preds"][i] < 0.5 or label_ind != label:
                    continue

                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                if len(contours) > 1 or len(contours) == 0 or contours[0].shape[0] < 3:
                    print(label, i, len(contours))
                    continue

                cv2.drawContours(image, contours, -1, rand_color(), -1)
                mask_data["output"]["shapes"].append(
                    {
                        "points": contours[0].reshape(-1, 2).tolist(),
                        "label": cfg[args.model].label[label_ind],
                        "score": mask_data["iou_preds"][i].item(),
                        "shape_type": "polygon",
                        "description": f"{i}",
                        "mask": None,
                        "flags": {},
                        "group_id": None,
                    }
                )

            save_path = img_path.replace(".png", f"_{model_suffix}_{label}.png")
            cv2.imwrite(save_path, image[:, :, ::-1])
            print(f"save res to {save_path}")

        json_save_path = img_path.replace(".png", f"_{model_suffix}.json")
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(mask_data["output"], f, indent=4, ensure_ascii=False)

    return mask_data


def main(args):
    mode = args.mode
    img_root = f"datasets/trop2/{mode}/JPEGImages"
    json_root = f"datasets/trop2/{mode}/jsons"
    img_list = glob.glob(os.path.join(img_root, "**/*.png"), recursive=True)
    json_list = glob.glob(os.path.join(json_root, "**/*.json"), recursive=True)
    img_list.sort()
    json_list.sort()
    # 计算指标
    metrics = {key: [] for key in cfg[args.model].label}
    for img_path, json_path in tqdm(
        zip(img_list[:], json_list[:]), total=len(img_list)
    ):
        if img_path.split("/")[-2] != json_path.split("/")[-2]:
            print(img_path, json_path)
            continue
        for key in metrics.keys():
            metrics[key].append(evaluate(img_path, json_path, keys=[key]))
    for key in metrics.keys():
        metrics[key] = np.nanmean(metrics[key], axis=0)
    print(metrics)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--save_res", action="store_true")
    parser.add_argument("--img_path", type=str, default="assets/0000.png")
    parser.add_argument(
        "--model", type=str, default="bplus_menu", help=", ".join(list(config.keys()))
    )

    args = parser.parse_args()
    load_model(args)
    if args.eval:
        main(args)
    else:
        infer(args.img_path, args.save_res)


# python infer.py --img_path assets/2025_10_30_10_55_56_511990_103761_29646.png --save_res --model bplus_me
