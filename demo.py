import json
import os

import cv2
import numpy as np


path = "assets/0001.png"
img = cv2.imread(path)
# me = cv2.imread(path.replace(".png", "_0.png"))
# nu = cv2.imread(path.replace(".png", "_1.png"))

colors = {"nu": "#1E386B", "me": "#11F433", "rg": "#FF00FF", "ct": "#fca60b"}
#fca60b

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))[::-1]


# me_color = me.astype(np.bool_) * np.array(hex_to_rgb(colors["me"]), dtype=np.uint8)
# nu_color = nu.astype(np.bool_) * np.array(hex_to_rgb(colors["nu"]), dtype=np.uint8)
# cv2.imwrite(path.replace(".png", "_me.png"), me_color)
# cv2.imwrite(path.replace(".png", "_nu.png"), nu_color)
# add_me_nu = (me_color&~nu) + nu_color
# cv2.imwrite(path.replace(".png", "_add.png"), add_me_nu)
scale = 1
with open(path.replace(".png", "_res.json"), "r") as f:
    data = json.load(f)
mask = np.zeros((img.shape[0]//scale, img.shape[1]//scale, 3), dtype=np.uint8)
for d in data["shapes"]:
    region = (np.array(d["points"])/scale).astype(np.int32)
    if d["label"] == 0:
        if region.shape[0] > 5:
            cv2.fillPoly(mask, [region], hex_to_rgb(colors["me"]))
            cv2.drawContours(mask, [region], -1, hex_to_rgb("#FF0000"), 2)
for d in data["shapes"]:
    region = (np.array(d["points"])/scale).astype(np.int32)            
    if d["label"] == 1:
        # print(1)
        if region.shape[0] > 5 and mask[region[:, 1], region[:, 0], 0].sum() != 0: # 过滤只有nu的区域
            cv2.fillPoly(mask, [region], hex_to_rgb(colors["nu"]))
mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
# _mask = (mask.sum(axis=2) == 0).astype(np.uint8)
# img = img * _mask[:,:,None]
_mask = (mask.sum(-1)>0).astype(np.uint8)
_mask1 = cv2.dilate(_mask, np.ones((3,3)), iterations=10) - _mask
contours, _ = cv2.findContours(_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask1 = np.zeros_like(mask)
for c in contours:
    cv2.fillPoly(mask1, [c], hex_to_rgb(colors["ct"]))
mask1 = mask1 * (~_mask.astype(np.bool_))[:, :, None]
mask = mask1+mask
# rgb = np.array(hex_to_rgb(colors["ct"]), dtype=np.uint8)
# mask = np.concatenate([_mask[:,:,None] ]*3, axis=2)*rgb + mask1

prob_cmp = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)
# prob_cmp[:, :, 3] = 127
cv2.imwrite("a4.png", mask)
# print(9)
