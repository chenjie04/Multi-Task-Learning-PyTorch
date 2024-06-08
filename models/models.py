#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTaskModel(nn.Module):
    """ Single-task baseline model with encoder + decoder """
    def __init__(self, backbone: nn.Module, decoder: nn.Module, task: str):
        super(SingleTaskModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder 
        self.task = task

    def forward(self, x):
        out_size = x.size()[2:]
        representation = self.backbone(x)
        out = self.decoder(representation)
        return {self.task: F.interpolate(out, out_size, mode='bilinear')}


class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list):
        super(MultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks # ['semseg', 'human_parts', 'sal', 'normals']
        
    def forward(self, x, feature_extraction=False):
        
        out_size = x.size()[2:]
        feature_maps, shared_representation = self.backbone(x)
        if not feature_extraction:
            return {task: F.interpolate(self.decoders[task](shared_representation), out_size, mode='bilinear') for task in self.tasks}
        else:
            return feature_maps
    
class SquadNetMultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list, num_shared: int):
        super(SquadNetMultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.num_shared = num_shared
        self.num_tasks = len(tasks)
        self.tasks = tasks # ['semseg', 'human_parts', 'sal', 'normals']
        self.tasks_id = {}
        id = 0
        for task in self.tasks:
            self.tasks_id[task] = id
            id = id + 1

    def forward(self, x, feature_extraction=False):
        if not feature_extraction:
            out_size = x.size()[2:]
            shared_representation = self.backbone(x)
            shared_representation = torch.chunk(shared_representation, self.num_tasks + self.num_shared, dim=1)
            shared_channels = torch.cat(shared_representation[-self.num_shared:], dim=1)
            return {task: F.interpolate(self.decoders[task](torch.cat([shared_representation[self.tasks_id[task]],shared_channels],dim=1)), out_size, mode='bilinear') for task in self.tasks}
        else:
            return self.backbone(x, feature_extraction)