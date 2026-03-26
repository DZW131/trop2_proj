import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# 将近取的点的个数设为10
NUMS = 10
config = {"tiny": {"ckpt_path": "checkpoints/sam2.1_hiera_tiny.pt", "config_file": "configs/sam2.1/sam2.1_hiera_t.yaml"},
          "large": {"ckpt_path": "checkpoints/sam2.1_hiera_large.pt", "config_file": "configs/sam2.1/sam2.1_hiera_l.yaml"}}

predictor = SAM2ImagePredictor(build_sam2(**config["tiny"]))
img = Image.open("truck.jpg")
image = np.array(img.copy().convert("RGB"))
predictor.set_image(image)
point_coords = np.array([[550,550]])
point_labels = np.array([1])
masks, scores, logits = predictor.predict(point_coords, point_labels, multimask_output=False)
print(masks.max())
cv2.imwrite("a.jpg", (masks[0]*255).astype(np.uint8))
