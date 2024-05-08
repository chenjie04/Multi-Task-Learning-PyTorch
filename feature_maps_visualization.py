import logging
from math import ceil
import torch
from utils.common_config import get_model, get_transformations, get_val_dataloader, get_val_dataset
from utils.config import create_config

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

config_env = 'configs/env.yml'
config_exp = 'configs/pascal/groupnet/groupnet_base.yml'
checkpoint_file = 'work_dirs/PASCALContext/groupnet/groupnet_base/best_model.pth.tar'
# config_exp = 'configs/pascal/groupnet_ablation/groupnet_tiny.yml'
# checkpoint_file = 'work_dirs/PASCALContext/groupnet/groupnet_tiny_1/best_model.pth.tar'
# config_exp = 'configs/pascal/groupnet_ablation/multi_task_baseline.yml'
# checkpoint_file = 'work_dirs/PASCALContext/groupnet/multi_task_baseline/best_model.pth.tar'

p = create_config(config_env, config_exp)
groupnet = get_model(p)
checkpoint = torch.load(checkpoint_file)
weights_dict = {}
for k, v in checkpoint.items():
    new_k = k.replace('module.', '') if 'module' in k else k
    weights_dict[new_k] = v

groupnet.load_state_dict(weights_dict, strict=True)
groupnet.cuda()
groupnet.eval()

p['valBatch'] = 1
_, val_transforms = get_transformations(p)
val_dataset = get_val_dataset(p, val_transforms)

sample = val_dataset.__getitem__(483) # {'image': '2008_001843', 'im_size': (375, 500)}

tsne = TSNE(n_components=2)

inputs, meta = sample['image'].unsqueeze(0).cuda(non_blocking=True), sample['meta']
img_size = (inputs.size(2), inputs.size(3))
features, output = groupnet(inputs, feature_extraction=True)
print(output.keys())
print(output['semseg'].shape)
print(output['human_parts'].shape)
print(output['sal'].shape)
print(output['normals'].shape)

for i in range(len(features)):
    print(features[i].shape)

class BilinearInterpolation(object):
    def __init__(self, w_rate: float, h_rate: float, *, align='center'):
        if align not in ['center', 'left']:
            logging.exception(f'{align} is not a valid align parameter')
            align = 'center'
        self.align = align
        self.w_rate = w_rate
        self.h_rate = h_rate

    def set_rate(self,w_rate: float, h_rate: float):
        self.w_rate = w_rate    # w 的缩放率
        self.h_rate = h_rate    # h 的缩放率

    # 由变换后的像素坐标得到原图像的坐标    针对高
    def get_src_h(self, dst_i,source_h,goal_h) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_i = float(dst_i * (source_h/goal_h))
        elif self.align == 'center':
            # 将两个图像的几何中心重合。
            src_i = float((dst_i + 0.5) * (source_h/goal_h) - 0.5)
        src_i += 0.001
        src_i = max(0.0, src_i)
        src_i = min(float(source_h - 1), src_i)
        return src_i
    # 由变换后的像素坐标得到原图像的坐标    针对宽
    def get_src_w(self, dst_j,source_w,goal_w) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_j = float(dst_j * (source_w/goal_w))
        elif self.align == 'center':
            # 将两个图像的几何中心重合。
            src_j = float((dst_j + 0.5) * (source_w/goal_w) - 0.5)
        src_j += 0.001
        src_j = max(0.0, src_j)
        src_j = min((source_w - 1), src_j)
        return src_j

    def transform(self, img):
        source_h, source_w, source_c = img.shape  # (235, 234, 3)
        goal_h, goal_w = round(
            source_h * self.h_rate), round(source_w * self.w_rate)
        new_img = np.zeros((goal_h, goal_w, source_c), dtype=np.uint8)

        for i in range(new_img.shape[0]):       # h
            src_i = self.get_src_h(i,source_h,goal_h)
            for j in range(new_img.shape[1]):
                src_j = self.get_src_w(j,source_w,goal_w)
                i2 = ceil(src_i)
                i1 = int(src_i)
                j2 = ceil(src_j)
                j1 = int(src_j)
                x2_x = j2 - src_j
                x_x1 = src_j - j1
                y2_y = i2 - src_i
                y_y1 = src_i - i1
                new_img[i, j] = img[i1, j1]*x2_x*y2_y + img[i1, j2] * \
                    x_x1*y2_y + img[i2, j1]*x2_x*y_y1 + img[i2, j2]*x_x1*y_y1
        return new_img

def visualize_feature_map(img_batch,out_path,type,BI):
    feature_map = torch.squeeze(img_batch)
    feature_map = feature_map.detach().cpu().numpy()

    feature_map_sum = feature_map[0, :, :]
    feature_map_sum = np.expand_dims(feature_map_sum, axis=2)
    for i in range(0, 768):
        feature_map_split = feature_map[i,:, :]
        feature_map_split = np.expand_dims(feature_map_split,axis=2)
        if i > 0:
            feature_map_sum +=feature_map_split
        feature_map_split = BI.transform(feature_map_split)

        plt.imshow(feature_map_split)
        plt.savefig(out_path + str(i) + "_{}.jpg".format(type) )
        plt.xticks()
        plt.yticks()
        plt.axis('off')

    feature_map_sum = BI.transform(feature_map_sum)
    plt.imshow(feature_map_sum)
    plt.savefig(out_path + "sum_{}.jpg".format(type))
    print("save sum_{}.jpg".format(type))


# stage_3 = features[3][0]
# stage_3_list = stage_3.split(128, dim=0)

BI = BilinearInterpolation(8, 8)
save_path = "feature_maps/"
visualize_feature_map(features[-1], save_path, "drone", BI)

"""
作者：CV技术指南
链接：https://juejin.cn/post/7072944829499441183
来源：稀土掘金
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
"""
