#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-26

from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import models

from .resnet import _ConvBnReLU, _ResLayer, _Stem


class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)


class DeepLabV3(nn.Sequential):
    """
    DeepLab v3: Dilated ResNet with multi-grid + improved ASPP
    """

    def __init__(self, n_classes, n_blocks, atrous_rates, multi_grids, output_stride):
        super(DeepLabV3, self).__init__()

        # Stride and dilation
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0]))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1]))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2]))
        self.add_module(
            "layer5", _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids)
        )
        self.add_module("aspp", _ASPP(ch[5], 256, atrous_rates))
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1))
        self.add_module("fc2", nn.Conv2d(256, n_classes, kernel_size=1))

class _ASPP2(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP2, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])

class DeepLabV3Py_ResNet101(nn.Module):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """


    def __init__(self, pretrain):
        super(DeepLabV3Py_ResNet101, self).__init__()
        atrous_rates = [6, 12, 18, 24]
        model = models.resnet101(pretrained=pretrain)


        # Layer 3 (OS=16 -> OS=8)
        model.layer3[0].conv2.stride = (1, 1)
        model.layer3[0].downsample[0].stride = (1, 1)
        for m in model.layer3[1:]:
            m.conv2.padding = (2, 2)
            m.conv2.dilation = (2, 2)

        # Layer 4 (OS=32 -> OS=8)
        model.layer4[0].conv2.stride = (1, 1)
        model.layer4[0].downsample[0].stride = (1, 1)
        for m in model.layer4[1:]:
            m.conv2.padding = (4, 4)
            m.conv2.dilation = (4, 4)

        # Remove "avgpool" and "fc", and add ASPP
        model = list(model.named_children())[:-2]
        model += [("aspp", _ASPP(2048, 256, [6, 12, 18, 24]))]
        concat_ch = 256 * (len(atrous_rates) + 2)
        model += [("fc1", _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1))]
        model += [("fc2", nn.Conv2d(256, 21, kernel_size=1))]
        self.model2 = nn.Sequential(OrderedDict(model))

    def freeze_bn(self):
        for n, m in self.named_modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()

    def freeze_layer(self):
        for n, m in self.named_parameters():
            break

    def forward(self, x):
        out = self.model2(x)
        return out


if __name__ == "__main__":
    model = DeepLabV3(
        n_classes=21,
        n_blocks=[3, 4, 23, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=8,
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
