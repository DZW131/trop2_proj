import json

import cv2
import numpy as np
import torch
from infer_utils import get_prompt
from PIL import Image
from sam2.build_sam import build_sam2
from torch.nn import functional as F
from torchvision.transforms import Normalize, ToTensor
from training.model.sam2 import SAM2Export
from utils import rand_color, to_json

model: SAM2Export = build_sam2(
    "configs/sam2.1/sam2.1_hiera_b+.yaml",
    ckpt_path="checkpoints/1118_me_nu2/checkpoints/checkpoint.pt",
    hydra_overrides_extra=["model._target_=training.model.sam2.SAM2Export"],
)

# print(model)
model.eval()
model = model.cuda()


def infer():
    img_path = "assets/0001.png"
    prompts_json_path = img_path.replace(".png", ".json")
    img = Image.open(img_path)
    img_tensor = (ToTensor()(img))[
        None, ...
    ].cuda()  # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data = {
        "key": [
            item["points"]
            for item in json.load(open(prompts_json_path))["shapes"]
            if item["label"] == "肿瘤细胞膜"
        ]
    }
    prompts = {}
    prompts = get_prompt(data, prompts, "key")
    ptl = np.concatenate(prompts["key"], axis=0)
    point_coords = torch.tensor(ptl[:, None, :2]).cuda()
    point_labels = torch.tensor(ptl[:, 2:]).cuda()
    print(point_coords.shape, point_labels.shape, img_tensor.shape)
    import time

    start = time.time()
    times = 1
    with torch.no_grad():
        for i in range(times):
            masks, iou_predictions = model(img_tensor, point_coords, point_labels)

    end = time.time()
    print(f"time: {(end - start) / (times) * 1000} ms")
    print(masks.shape)
    print(iou_predictions.shape)
    print([*img_tensor.shape[-2:]])
    masks = (
        F.interpolate(
            masks.to(torch.float32),
            [*img_tensor.shape[-2:]],
            mode="bilinear",
            align_corners=False,
        )
        > 0
    )
    data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imageData": None,
        "imageHeight": 1024,
        "imageWidth": 1024,
        "imagePath": "2025_10_30_10_55_56_511990_17297_93898.png",
    }
    for i in range(masks.shape[1]):
        mask = masks[:, i : i + 1].squeeze(1).cpu().numpy()
        shapes = to_json(mask, i)
        data["shapes"].extend(shapes)
        mask = mask.sum(0)
        mask = mask * (255 // mask.max())
        mask = mask.astype(np.bool_)
        cv2.imwrite(img_path.replace(".png", f"_{i}.png"), mask.astype(np.uint8) * 255)
    json.dump(
        data,
        open(prompts_json_path.replace(".json", "_res.json"), "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )
    # image = np.array(img).copy()
    # for i, mask in enumerate(masks.cpu().numpy()):
    #     cv2.imwrite(f"temp/a_{i}.jpg", mask.astype(np.uint8)*255)
    #     cv2.drawContours(image, cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0], -1, rand_color(), -1)
    # cv2.imwrite(img_path.replace(".png", "_res.png"), image[:,:,::-1])


def export():
    input_img = torch.randn(1, 3, 1024, 1024).cuda()
    point_coords = torch.randn(4, 1, 2).cuda()
    point_labels = torch.randn(4, 1).cuda()
    torch.onnx.export(
        model,
        (input_img, point_coords, point_labels),
        "sam2_mul_output.onnx",
        opset_version=17,
        verbose=True,
        input_names=["image", "point_coords", "point_labels"],
        output_names=["low_res_masks", "iou_predictions"],
        dynamic_axes={
            # "image": {0: "batch_size"},
            "point_coords": {0: "batch_size"},
            "point_labels": {0: "batch_size"},
            "low_res_masks": {0: "batch_size"},
            "iou_predictions": {0: "batch_size"},
        },
    )


def export_encoder():
    input_img = torch.randn(1, 3, 1024, 1024).cuda()
    # model.forward_encoder(input_img)
    model.forward = model.forward_encoder
    torch.onnx.export(
        model,
        input_img,
        "sam2.encoder.onnx",
        opset_version=17,
        verbose=False,
        input_names=["image"],
        output_names=["high_res_feats_0", "high_res_feats_1", "image_embed"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "high_res_feats_0": {0: "batch_size"},
            "high_res_feats_1": {0: "batch_size"},
            "image_embed": {0: "batch_size"},
        },
    )


def export_decoder():
    model.to("cpu")
    input_img = torch.randn(1, 3, 1024, 1024)
    high_res_feats_0, high_res_feats_1, image_embed = model.forward_encoder(input_img)
    point_coords = torch.randn(2, 1, 2)
    point_labels = torch.randn(2, 1)
    # mask_input = torch.randn(1, 1, 256, 256)
    # has_mask_input = torch.randn(1, 1)
    model.forward = model.forward_decoder
    torch.onnx.export(
        model,
        (
            image_embed.to("cpu"),
            high_res_feats_0.to("cpu"),
            high_res_feats_1.to("cpu"),
            point_coords.to("cpu"),
            point_labels.to("cpu"),
            # mask_input.to("cpu"),
            # has_mask_input.to("cpu"),
        ),
        "sam2.decoder.onnx",
        opset_version=17,
        verbose=False,
        input_names=[
            "image_embed",
            "high_res_feats_0",
            "high_res_feats_1",
            "point_coords",
            "point_labels",
            # "mask_input",
            # "has_mask_input",
        ],
        output_names=["low_res_masks", "iou_predictions"],
        dynamic_axes={
            "point_coords": {0: "num_points"},
            "point_labels": {0: "num_points"},
            "low_res_masks": {0: "num_points"},
            "iou_predictions": {0: "num_points"},
        },
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--export", action="store_true")
    parser.add_argument("-i", "--infer", action="store_true")
    parser.add_argument("-ep", "--export_partition", action="store_true")

    args = parser.parse_args()
    if args.infer:
        infer()
    if args.export:
        export()
    if args.export_partition:
        # export_encoder()
        export_decoder()
