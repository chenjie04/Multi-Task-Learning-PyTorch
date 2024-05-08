"""
# ------------------------------------------------------------
# Version :   1.0
# Author  :   chenjie04
# Email   :   gxuchenjie04@gmail.com
# Time    :   2024/03/07 15:15:56
# Descript:   读取某张图片（如2008_000195.png）的human_parts的标注
#             并将其转换成彩色图像
# ---------------------------------------------------------------
"""

import os
from PIL import Image
import numpy as np
from data.pascal_context import PASCALContext

np.set_printoptions(threshold=np.inf)


SEG_EXAMPLE = (
    "/home/chenjie04/workstation/data/PASCAL_MT/semseg/pascal-context/2008_000195.png"
)
gt_img = Image.open(SEG_EXAMPLE)
gt_palette = gt_img.getpalette()


ORI_PATH = "/home/chenjie04/workstation/data/PASCAL_MT/human_parts/"
SAVE_PATH = "./result_visualization/human_parts_gt_vis"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

dataset = PASCALContext(
    root="/home/chenjie04/workstation/data/PASCAL_MT/",
    split=["val"],
    do_edge=False,
    do_human_parts=True,
)

IMG_NAME = "2008_000834"

index = dataset.im_ids.index(IMG_NAME)
sample = dataset[index]
print(sample.keys())

img = np.array(sample["human_parts"]).astype(np.int8)  # 如果没有转成int8，结果会一片黑
img = Image.fromarray(img, mode="P")
img.putpalette(gt_palette)
img.save(os.path.join(SAVE_PATH, f"{IMG_NAME}.png"))
