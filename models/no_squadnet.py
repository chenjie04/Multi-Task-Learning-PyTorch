import math
from typing import List, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_
from mmdet.utils import ConfigType
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from mmengine.utils import digit_version, is_tuple_of
from mmdet.utils import MultiConfig, OptConfigType, OptMultiConfig
from timm.models.layers import DropPath


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class StemLayer(BaseModule):
    r"""Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        embed_dim (int): number of output channels
        num_tasks (int): number of tasks
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(
        self,
        in_chans=3,
        embed_dim=32,
        num_tasks=5,
        num_shared=1,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="SyncBN"),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.num_tasks = num_tasks
        self.embeddings = ConvModule(
            in_chans,
            embed_dim * (num_tasks + num_shared),
            kernel_size=3,
            stride=2,
            padding=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
        )
        self.project = nn.Sequential(
            DepthwiseSeparableConvModule(
                embed_dim * (num_tasks + num_shared),
                embed_dim * (num_tasks + num_shared),
                kernel_size=3,
                padding=1,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
            )
        )

    def forward(self, x):
        out = self.embeddings(x)
        out = self.project(out)
        return out


class DownsampleLayer(BaseModule):
    r"""Downsample layer of InternImage
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(
        self,
        channels,
        num_tasks=4,
        num_shared=1,
        channel_shuffle=True,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="GELU"),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_tasks = num_tasks
        self.num_shared = num_shared
        self.channel_shuffle = channel_shuffle
        self.total_channels = channels * (num_tasks + num_shared)
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                self.total_channels,
                self.total_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=self.total_channels,
                bias=False,
            ),
            build_norm_layer(norm_cfg, self.total_channels)[1],
            nn.Conv2d(
                self.total_channels,
                self.total_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            build_norm_layer(norm_cfg, self.total_channels)[1],
            build_activation_layer(act_cfg),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                self.total_channels,
                self.total_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            build_norm_layer(norm_cfg, self.total_channels)[1],
            build_activation_layer(act_cfg),
            nn.Conv2d(
                self.total_channels,
                self.total_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=self.total_channels,
                bias=False,
            ),
            build_norm_layer(norm_cfg, self.total_channels)[1],
            nn.Conv2d(
                self.total_channels,
                self.total_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            build_norm_layer(norm_cfg, self.total_channels)[1],
            build_activation_layer(act_cfg),
        )

        self.feed_forward = DwFeadForward(
            self.total_channels * 2,
            mlp_ratio=1.0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x1, x2], dim=1)
        if self.channel_shuffle:
            x = channel_shuffle(x, (self.num_tasks + self.num_shared) * 2)
        x = self.feed_forward(x)
        return x


class GConvModule(BaseModule):
    def __init__(
        self,
        channels: int = 32,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        pad: int = 1,
        norm_cfg: ConfigType = dict(type="SyncBN"),
        act_cfg: ConfigType = dict(type="GELU"),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            build_norm_layer(norm_cfg, channels)[1],
            build_activation_layer(act_cfg),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            build_norm_layer(norm_cfg, channels)[1],
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            build_norm_layer(norm_cfg, channels)[1],
            build_activation_layer(act_cfg),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class DwFeadForward(nn.Module):
    def __init__(
        self,
        in_channel: int = 64,
        mlp_ratio: float = 1.0,
        drop_rate: float = 0.1,
        norm_cfg: ConfigType = dict(type="SyncBN"),
        act_cfg: ConfigType = dict(type="GELU"),
    ):
        super(DwFeadForward, self).__init__()
        self.in_channel = in_channel
        self.mlp_ratio = mlp_ratio
        self.hidden_fetures = int(in_channel * mlp_ratio)

        self.feedforward = nn.Sequential(
            nn.Conv2d(
                in_channel,
                self.hidden_fetures,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            build_norm_layer(norm_cfg, in_channel)[1],
            build_activation_layer(act_cfg),
            nn.Conv2d(
                self.hidden_fetures,
                self.hidden_fetures,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=self.hidden_fetures,
                bias=False,
            ),
            build_norm_layer(norm_cfg, self.hidden_fetures)[1],
            nn.Conv2d(
                self.hidden_fetures,
                self.hidden_fetures,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            build_norm_layer(norm_cfg, self.hidden_fetures)[1],
            build_activation_layer(act_cfg),
        )

    def forward(self, x):
        """

        :param input: [bs, C, H, W]
        :return: [bs, C, H, W]
        """

        # feed forward
        x = self.feedforward(x)
        return x


class NoGroupConvModule(BaseModule):
    def __init__(
        self,
        channels: int = 32,
        num_tasks: int = 5,
        num_shared: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        pad: int = 1,
        mlp_ratio: float = 1.0,
        channel_shuffle: bool = True,
        norm_cfg: ConfigType = dict(type="SyncBN"),
        act_cfg: ConfigType = dict(type="GELU"),
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.num_tasks = num_tasks
        self.num_shared = num_shared
        self.channel_shuffle = channel_shuffle

        total_channels = channels * (num_tasks + num_shared)
        self.conv = GConvModule(
            channels=total_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            pad=pad,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.feed_forward = DwFeadForward(
            int(channels * (num_tasks + num_shared)),
            mlp_ratio=mlp_ratio,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x):
        # shortcut = x
        x = self.conv(x)
        if self.channel_shuffle:
            x = channel_shuffle(x, self.num_tasks + self.num_shared)
        x = self.feed_forward(x)
        # x = (
        #     shortcut + x
        # )
        return x


class BasicBlock(BaseModule):
    r"""Basic Block
    Args:
        channels (int): number of input channels
        num_tasks (int): number of tasks
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
    """

    def __init__(
        self,
        channels=32,
        num_tasks=5,
        num_shared=1,
        depth=2,
        channel_shuffle=True,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        pad: int = 1,
        downsample=True,
        mlp_ratio=1.0,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="SyncBN"),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.channels = channels
        self.num_tasks = num_tasks
        self.num_shared = num_shared
        self.depth = depth

        self.blocks = nn.ModuleList(
            NoGroupConvModule(
                channels=channels,
                num_tasks=num_tasks,
                num_shared=num_shared,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                pad=pad,
                mlp_ratio=mlp_ratio,
                channel_shuffle=channel_shuffle,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
            )
            for i in range(depth)
        )

        self.downsample = (
            DownsampleLayer(
                channels=channels,
                num_tasks=num_tasks,
                num_shared=num_shared,
                channel_shuffle=channel_shuffle,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
            )
            if downsample
            else None
        )

    def forward(self, x, return_wo_downsample=False):
        for blk in self.blocks:
            x = blk(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x


class NoSquadNet(BaseModule):
    r"""
    Args:
        core_op (str): Core operator. Default: 'DCNv3'
        channels (int): Number of the first stage. Default: 128
        depths (list): Depth of each block. Default: [8, 8, 4]
        groups (list): Groups of each block. Default: [4, 8, 16]
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 1.
        drop_rate (float): Probability of an element to be zeroed. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        act_layer (str): Activation layer. Default: 'GELU'
        norm_layer (str): Normalization layer. Default: 'LN'
        layer_scale (bool): Whether to use layer scale. Default: False
    """

    def __init__(
        self,
        core_op="SeparableGroupConv",
        channels=32,
        num_tasks=4,
        num_shared=1,
        depths=[2, 4, 2],
        channel_shuffle=[True, True, True],
        mlp_ratios=[1.0, 1.0, 1.0],
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="SyncBN"),
        out_indices=(2,),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_tasks = num_tasks
        self.num_features = int(channels * 2 ** (self.num_levels - 1))
        self.mlp_ratios = mlp_ratios
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        print(f"using core type: {core_op}")
        print(f"unmber of tasks: {num_tasks}")
        print(f"using activation layer: {act_cfg}")
        print(f"using main norm layer: {norm_cfg}")

        in_chans = 3
        self.patch_embed = StemLayer(
            in_chans=in_chans,
            embed_dim=channels,
            num_tasks=num_tasks,
            num_shared=num_shared,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
        )

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = BasicBlock(
                channels=int(channels * 2**i),
                num_tasks=num_tasks,
                num_shared=num_shared,
                depth=depths[i],
                channel_shuffle=channel_shuffle[i],
                mlp_ratio=self.mlp_ratios[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                downsample=(i < self.num_levels - 1),
            )
            self.levels.append(level)

        self.num_layers = len(depths)

    def forward(self, x):
        x = self.patch_embed(x)

        seq_out = []
        for level_idx, level in enumerate(self.levels):
            x, x_ = level(x, return_wo_downsample=True)
            if level_idx in self.out_indices:
                seq_out.append(x_.contiguous())

        return seq_out[-1]


def test():
    x = torch.randn(1, 3, 512, 512).cuda()
    model = NoSquadNet().cuda()
    y = model(x)
    print(model)
    print(y.shape)

    import time

    time.sleep(10)


if __name__ == "__main__":
    test()
