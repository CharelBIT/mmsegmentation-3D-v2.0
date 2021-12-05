# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections import OrderedDict
from typing import Callable, Sequence, Type, Union

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
# from monai.networks.layers.factories import Conv, Dropout, Norm, Pool
# from ..utils.layer_factories import Conv, Dropout, Norm, Pool
from mmcv.cnn import ConvModule

__all__ = [
    "DenseNet",
    "densenet",
    "Densenet",
    "DenseNet121",
    "densenet121",
    "Densenet121",
    "DenseNet169",
    "densenet169",
    "Densenet169",
    "DenseNet201",
    "densenet201",
    "Densenet201",
    "DenseNet264",
    "densenet264",
    "Densenet264",
]


class _DenseLayer(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, growth_rate: int, bn_size: int, dropout_prob: float
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            growth_rate: how many filters to add each layer (k in paper).
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            dropout_prob: dropout rate after each dense layer.
        """
        super(_DenseLayer, self).__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", norm_type(in_channels))
        self.layers.add_module("relu1", nn.ReLU(inplace=True))
        self.layers.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))

        self.layers.add_module("norm2", norm_type(out_channels))
        self.layers.add_module("relu2", nn.ReLU(inplace=True))
        self.layers.add_module("conv2", conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self, spatial_dims: int, layers: int, in_channels: int, bn_size: int, growth_rate: int, dropout_prob: float
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            layers: number of layers in the block.
            in_channels: number of the input channel.
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            growth_rate: how many filters to add each layer (k in paper).
            dropout_prob: dropout rate after each dense layer.
        """
        super(_DenseBlock, self).__init__()
        for i in range(layers):
            layer = _DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            out_channels: number of the output classes.
        """
        super(_Transition, self).__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module("norm", norm_type(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    Densenet based on: `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
        dropout_prob: dropout rate after each dense layer.
    """

    arch_settings = {
        121: (6, 12, 24, 16),
        169: (6, 12, 32, 32),
        201: (6, 12, 48, 32),
        264: (6, 12, 48, 32)
    }

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_channels=64,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style='pytorch',
        deep_stem=False,
        avg_down=False,
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        plugins=None,
        multi_grid=None,
        contract_dilation=False,
        with_cp=False,
        zero_init_residual=True):
        super(DenseNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for DenseNet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.deep_stem = deep_stem
        self._make_stem_layer(in_channels, stem_channels)
        self.inplanes = stem_channels
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        for i, num_layers in enumerate(self.stage_blocks):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module("norm5", norm_type(in_channels))
            else:
                _out_channels = in_channels // 2
                trans = _Transition(spatial_dims, in_channels=in_channels, out_channels=_out_channels)
                self.features.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels

        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", nn.ReLU(inplace=True)),
                    ("pool", avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)),
                    ("out", nn.Linear(in_channels, out_channels)),
                ]
            )
        )

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.class_layers(x)
        return x

    def _make_stem_layer(self, in_channels, stem_channels):
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


def _load_state_dict(model, arch, progress):
    """
    This function is used to load pretrained models.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.

    """
    model_urls = {
        "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
        "densenet169": "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
        "densenet201": "https://download.pytorch.org/models/densenet201-c1103571.pth",
    }
    if arch in model_urls:
        model_url = model_urls[arch]
    else:
        raise ValueError(
            "only 'densenet121', 'densenet169' and 'densenet201' are supported to load pretrained weights."
        )
    pattern = re.compile(
        r"^(.*denselayer\d+)(\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + ".layers" + res.group(2) + res.group(3)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    model_dict = model.state_dict()
    state_dict = {
        k: v for k, v in state_dict.items() if (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)
    }
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


class DenseNet121(DenseNet):
    """DenseNet121 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super(DenseNet121, self).__init__(
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            **kwargs,
        )
        if pretrained:
            # it only worked when `spatial_dims` is 2
            _load_state_dict(self, "densenet121", progress)


class DenseNet169(DenseNet):
    """DenseNet169 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 32, 32),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super(DenseNet169, self).__init__(
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            **kwargs,
        )
        if pretrained:
            # it only worked when `spatial_dims` is 2
            _load_state_dict(self, "densenet169", progress)


class DenseNet201(DenseNet):
    """DenseNet201 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 48, 32),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super(DenseNet201, self).__init__(
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            **kwargs,
        )
        if pretrained:
            # it only worked when `spatial_dims` is 2
            _load_state_dict(self, "densenet201", progress)


class DenseNet264(DenseNet):
    """DenseNet264"""

    def __init__(
        self,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 48, 32),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super(DenseNet264, self).__init__(
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            **kwargs,
        )
        if pretrained:
            raise NotImplementedError("Currently PyTorch Hub does not provide densenet264 pretrained models.")


Densenet = densenet = DenseNet
Densenet121 = densenet121 = DenseNet121
Densenet169 = densenet169 = DenseNet169
Densenet201 = densenet201 = DenseNet201
Densenet264 = densenet264 = DenseNet264
