import torch
from utils.common_config import get_model, get_transformations, get_val_dataloader, get_val_dataset
from utils.config import create_config

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

config_env = 'configs/env.yml'
groupnet_config_exp = 'configs/pascal/groupnet/groupnet_base.yml'
groupnet_checkpoint_file = 'work_dirs/PASCALContext/groupnet/groupnet_base/best_model.pth.tar'
# groupnet_tiny_config_exp = 'configs/pascal/groupnet_ablation/groupnet_tiny.yml'
# groupnet_tiny_checkpoint_file = 'work_dirs/PASCALContext/groupnet/groupnet_tiny_1/best_model.pth.tar'
mtl_config_exp = 'configs/pascal/resnet18/multi_task_baseline.yml'
mtl_checkpoint_file = 'work_dirs/PASCALContext/resnet18/multi_task_baseline/best_model.pth.tar'

groupnet_p = create_config(config_env, groupnet_config_exp)
groupnet = get_model(groupnet_p)
groupnet_checkpoint = torch.load(groupnet_checkpoint_file)
weights_dict = {}
for k, v in groupnet_checkpoint.items():
    new_k = k.replace('module.', '') if 'module' in k else k
    weights_dict[new_k] = v

groupnet.load_state_dict(weights_dict, strict=True)
groupnet.cuda()
groupnet.eval()

mtl_p = create_config(config_env, mtl_config_exp)
mtl = get_model(mtl_p)
mtl_checkpoint = torch.load(mtl_checkpoint_file)
weights_dict = {}
for k, v in mtl_checkpoint.items():
    new_k = k.replace('module.', '') if 'module' in k else k
    weights_dict[new_k] = v

mtl.load_state_dict(weights_dict, strict=True)
mtl.cuda()
mtl.eval()

groupnet_p['valBatch'] = 1
_, val_transforms = get_transformations(groupnet_p)
val_dataset = get_val_dataset(groupnet_p, val_transforms)

sample = val_dataset.__getitem__(483) # {'image': '2008_001843', 'im_size': (375, 500)}
# sample = val_dataset.__getitem__(2)

tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=2023)

inputs, meta = sample['image'].unsqueeze(0).cuda(non_blocking=True), sample['meta']
img_size = (inputs.size(2), inputs.size(3))
groupnet_features = groupnet(inputs, feature_extraction=True)


mtl_features = mtl(inputs, feature_extraction=True)

for i in range(len(groupnet_features)):
    print(groupnet_features[i].shape)

for i in range(len(mtl_features)):
    print(mtl_features[i].shape)



plt.subplot(2, 2, 1)
groupnet_stem = groupnet_features[0][0]
groupnet_stem = tsne.fit_transform(groupnet_stem.view(192, -1).cpu().detach().numpy())
plt.scatter(groupnet_stem[:, 0], groupnet_stem[:, 1], color = '#14517c', s=3, label='SquadNet')


mtl_stem = mtl_features[0][0]
mtl_stem = tsne.fit_transform(mtl_stem.view(64, -1).cpu().detach().numpy())
plt.scatter(mtl_stem[:, 0], mtl_stem[:, 1], color = '#f3d266', s=3, label='ResNet-18')

plt.legend(loc="lower left", prop={'size': 6})
plt.title('Stem (Stage 0)', y=-0.3)

plt.subplot(2, 2, 2)
groupnet_stage_1 = groupnet_features[1][0]
groupnet_stage_1 = tsne.fit_transform(groupnet_stage_1.view(192, -1).cpu().detach().numpy())
plt.scatter(groupnet_stage_1[:, 0], groupnet_stage_1[:, 1], color = '#14517c', s=3, label='SquadNet')




mtl_stage_1 = mtl_features[1][0]
mtl_stage_1 = tsne.fit_transform(mtl_stage_1.view(128, -1).cpu().detach().numpy())
plt.scatter(mtl_stage_1[:, 0], mtl_stage_1[:, 1], color = '#f3d266', s=3, label='ResNet-18')

plt.legend(loc="lower left", prop={'size': 6})
plt.title('Stage 1', y=-0.3)

plt.subplot(2, 2, 3)
groupnet_stage_2 = groupnet_features[2][0]
groupnet_stage_2 = tsne.fit_transform(groupnet_stage_2.view(384, -1).cpu().detach().numpy())
plt.scatter(groupnet_stage_2[:, 0], groupnet_stage_2[:, 1], color = '#14517c', s=3, label='SquadNet')

mtl_stage_2 = mtl_features[2][0]
mtl_stage_2 = tsne.fit_transform(mtl_stage_2.view(256, -1).cpu().detach().numpy())
plt.scatter(mtl_stage_2[:, 0], mtl_stage_2[:, 1], color = '#f3d266', s=3, label='ResNet-18')

plt.legend(loc="lower left", prop={'size': 6})
plt.title('Stage 2', y=-0.3)


plt.subplot(2, 2, 4)
groupnet_stage_3 = groupnet_features[3][0]
groupnet_stage_3 = tsne.fit_transform(groupnet_stage_3.view(768, -1).cpu().detach().numpy())

plt.scatter(groupnet_stage_3[:, 0], groupnet_stage_3[:, 1], color = '#14517c', s=3, label='SquadNet')
# stage_3_list = np.split(stage_3, 6, axis=0)

# plt.scatter(stage_3_list[0][:, 0], stage_3_list[0][:, 1], color = '#14517c')
# plt.scatter(stage_3_list[1][:, 0], stage_3_list[1][:, 1], color = '#2f7fc1')
# plt.scatter(stage_3_list[2][:, 0], stage_3_list[2][:, 1], color = '#96c37d')
# plt.scatter(stage_3_list[3][:, 0], stage_3_list[3][:, 1], color = '#f3d266')
# plt.scatter(stage_3_list[4][:, 0], stage_3_list[4][:, 1], color = '#d8383a')
# plt.scatter(stage_3_list[5][:, 0], stage_3_list[5][:, 1], color = '#c497b2')

mtl_stage_3 = mtl_features[3][0]
mtl_stage_3 = tsne.fit_transform(mtl_stage_3.view(512, -1).cpu().detach().numpy())
plt.scatter(mtl_stage_3[:, 0], mtl_stage_3[:, 1], color = '#f3d266', s=3, label='ResNet-18')
plt.legend(loc="lower left", prop={'size': 6})
plt.title('Stage 3', y=-0.3)
        
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)


plt.savefig('channel_visualization_squadnet_resnet.pdf')
