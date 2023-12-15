from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

# from .DCNv2.dcn_v2 import DCN
from .non_local import NLBlockND
from .dcnv2 import DeformableConv2d
from .coatnet import coatnet_2
from .convtransformer import Transformer


import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .aspp import build_aspp
from .decoder_masks import build_decoder

webroot = 'https://tigress-web.princeton.edu/~fy/drn/models/'

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'drn-c-26': webroot + 'drn_c_26-ddedf421.pth',
    'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth',
    'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth',
    'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth',
    'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth',
    'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
    'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
}


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True, BatchNorm=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):

    def __init__(self, block, layers, arch='D',
                 channels=[16, 32, 64, 128, 256, 512, 512, 512],
                 BatchNorm=None):
        super(DRN, self).__init__()
        self.channels = channels
        self.inplanes = channels[0]
        self.out_dim = channels[-1]
        self.arch = arch

        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                                   padding=3, bias=False)
            self.bn1 = BatchNorm(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(
                BasicBlock, channels[0], layers[0], stride=1, BatchNorm=BatchNorm)
            self.layer2 = self._make_layer(
                BasicBlock, channels[1], layers[1], stride=2, BatchNorm=BatchNorm)

        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3,
                          bias=False),
                BatchNorm(channels[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_conv_layers(
                channels[0], layers[0], stride=1, BatchNorm=BatchNorm)
            self.layer2 = self._make_conv_layers(
                channels[1], layers[1], stride=2, BatchNorm=BatchNorm)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, BatchNorm=BatchNorm)
        self.layer5 = self._make_layer(block, channels[4], layers[4],
                                       dilation=2, new_level=False, BatchNorm=BatchNorm)
        self.layer6 = None if layers[5] == 0 else \
            self._make_layer(block, channels[5], layers[5], dilation=4,
                             new_level=False, BatchNorm=BatchNorm)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else \
                self._make_layer(BasicBlock, channels[6], layers[6], dilation=2,
                                 new_level=False, residual=False, BatchNorm=BatchNorm)
            self.layer8 = None if layers[7] == 0 else \
                self._make_layer(BasicBlock, channels[7], layers[7], dilation=1,
                                 new_level=False, residual=False, BatchNorm=BatchNorm)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(channels[6], layers[6], dilation=2, BatchNorm=BatchNorm)
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(channels[7], layers[7], dilation=1, BatchNorm=BatchNorm)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True, BatchNorm=None):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation), BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1, BatchNorm=None):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        low_level_feat = x

        x = self.layer4(x)
        x = self.layer5(x)

        if self.layer6 is not None:
            x = self.layer6(x)

        if self.layer7 is not None:
            x = self.layer7(x)

        if self.layer8 is not None:
            x = self.layer8(x)

        return x, low_level_feat


class DRN_A(nn.Module):

    def __init__(self, block, layers, BatchNorm=None):
        self.inplanes = 64
        super(DRN_A, self).__init__()
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilation=2, BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilation=4, BatchNorm=BatchNorm)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dilation=(dilation, dilation, ), BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def drn_a_50(BatchNorm, pretrained=True):
    model = DRN_A(Bottleneck, [3, 4, 6, 3], BatchNorm=BatchNorm)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def drn_c_26(BatchNorm, pretrained=True):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = model_zoo.load_url(model_urls['drn-c-26'])
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def drn_c_42(BatchNorm, pretrained=True):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = model_zoo.load_url(model_urls['drn-c-42'])
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def drn_c_58(BatchNorm, pretrained=True):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = model_zoo.load_url(model_urls['drn-c-58'])
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def drn_d_22(BatchNorm, pretrained=True):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = model_zoo.load_url(model_urls['drn-d-22'])
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def drn_d_24(BatchNorm, pretrained=True):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], arch='D', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = model_zoo.load_url(model_urls['drn-d-24'])
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def drn_d_38(BatchNorm, pretrained=True):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = model_zoo.load_url(model_urls['drn-d-38'])
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def drn_d_40(BatchNorm, pretrained=True):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = model_zoo.load_url(model_urls['drn-d-40'])
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def drn_d_54(BatchNorm=nn.BatchNorm2d, pretrained=False):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = model_zoo.load_url(model_urls['drn-d-54'])
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def drn_d_105(BatchNorm, pretrained=True):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = model_zoo.load_url(model_urls['drn-d-105'])
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model














BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

# def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
#     return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


# def conv3x3(in_planes, out_planes, stride=1):
#     "3x3 convolution with padding"
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)

# class convolution(nn.Module):
#     def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
#         super(convolution, self).__init__()

#         pad = (k - 1) // 2
#         self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
#         self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         conv = self.conv(x)
#         bn   = self.bn(conv)
#         relu = self.relu(bn)
#         return relu

# def make_cnv_layer(inp_dim, out_dim):
#     return convolution(3, inp_dim, out_dim)

# class BasicBlock(nn.Module):
#     def __init__(self, inplanes, planes, stride=1, dilation=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
#                                stride=stride, padding=dilation,
#                                bias=False, dilation=dilation)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=dilation,
#                                bias=False, dilation=dilation)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.stride = stride

#     def forward(self, x, residual=None):
#         if residual is None:
#             residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         out += residual
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 2

#     def __init__(self, inplanes, planes, stride=1, dilation=1):
#         super(Bottleneck, self).__init__()
#         expansion = Bottleneck.expansion
#         bottle_planes = planes // expansion
#         self.conv1 = nn.Conv2d(inplanes, bottle_planes,
#                                kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
#                                stride=stride, padding=dilation,
#                                bias=False, dilation=dilation)
#         self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv3 = nn.Conv2d(bottle_planes, planes,
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.stride = stride

#     def forward(self, x, residual=None):
#         if residual is None:
#             residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         out += residual
#         out = self.relu(out)

#         return out


# class BottleneckX(nn.Module):
#     expansion = 2
#     cardinality = 32

#     def __init__(self, inplanes, planes, stride=1, dilation=1):
#         super(BottleneckX, self).__init__()
#         cardinality = BottleneckX.cardinality
#         # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
#         # bottle_planes = dim * cardinality
#         bottle_planes = planes * cardinality // 32
#         self.conv1 = nn.Conv2d(inplanes, bottle_planes,
#                                kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
#                                stride=stride, padding=dilation, bias=False,
#                                dilation=dilation, groups=cardinality)
#         self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv3 = nn.Conv2d(bottle_planes, planes,
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.stride = stride

#     def forward(self, x, residual=None):
#         if residual is None:
#             residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         out += residual
#         out = self.relu(out)

#         return out


# class Root(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, residual):
#         super(Root, self).__init__()
#         self.conv = nn.Conv2d(
#             in_channels, out_channels, 1,
#             stride=1, bias=False, padding=(kernel_size - 1) // 2)
#         self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.residual = residual

#     def forward(self, *x):
#         children = x
#         x = self.conv(torch.cat(x, 1))
#         x = self.bn(x)
#         if self.residual:
#             x += children[0]
#         x = self.relu(x)

#         return x


# class Tree(nn.Module):
#     def __init__(self, levels, block, in_channels, out_channels, stride=1,
#                  level_root=False, root_dim=0, root_kernel_size=1,
#                  dilation=1, root_residual=False):
#         super(Tree, self).__init__()
#         if root_dim == 0:
#             root_dim = 2 * out_channels
#         if level_root:
#             root_dim += in_channels
#         if levels == 1:
#             self.tree1 = block(in_channels, out_channels, stride,
#                                dilation=dilation)
#             self.tree2 = block(out_channels, out_channels, 1,
#                                dilation=dilation)
#         else:
#             self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
#                               stride, root_dim=0,
#                               root_kernel_size=root_kernel_size,
#                               dilation=dilation, root_residual=root_residual)
#             self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
#                               root_dim=root_dim + out_channels,
#                               root_kernel_size=root_kernel_size,
#                               dilation=dilation, root_residual=root_residual)
#         if levels == 1:
#             self.root = Root(root_dim, out_channels, root_kernel_size,
#                              root_residual)
#         self.level_root = level_root
#         self.root_dim = root_dim
#         self.downsample = None
#         self.project = None
#         self.levels = levels
#         if stride > 1:
#             self.downsample = nn.MaxPool2d(stride, stride=stride)
#         if in_channels != out_channels:
#             self.project = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels,
#                           kernel_size=1, stride=1, bias=False),
#                 nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
#             )

#     def forward(self, x, residual=None, children=None):
#         children = [] if children is None else children
#         bottom = self.downsample(x) if self.downsample else x
#         residual = self.project(bottom) if self.project else bottom
#         if self.level_root:
#             children.append(bottom)
#         x1 = self.tree1(x, residual)
#         if self.levels == 1:
#             x2 = self.tree2(x1)
#             x = self.root(x2, x1, *children)
#         else:
#             children.append(x1)
#             x = self.tree2(x1, children=children)
#         return x


# class DLA(nn.Module):
#     def __init__(self, levels, channels, num_classes=1000,
#                  block=BasicBlock, residual_root=False, linear_root=False):
#         super(DLA, self).__init__()
#         self.channels = channels
#         self.num_classes = num_classes
#         self.base_layer = nn.Sequential(
#             nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
#                       padding=3, bias=False),
#             nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True))
#         self.level0 = self._make_conv_level(
#             channels[0], channels[0], levels[0])
#         self.level1 = self._make_conv_level(
#             channels[0], channels[1], levels[1], stride=2)
#         self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
#                            level_root=False,
#                            root_residual=residual_root)
#         self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
#                            level_root=True, root_residual=residual_root)
#         self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
#                            level_root=True, root_residual=residual_root)
#         self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
#                            level_root=True, root_residual=residual_root)


#     def _make_level(self, block, inplanes, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or inplanes != planes:
#             downsample = nn.Sequential(
#                 nn.MaxPool2d(stride, stride=stride),
#                 nn.Conv2d(inplanes, planes,
#                           kernel_size=1, stride=1, bias=False),
#                 nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
#             )

#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample=downsample))
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes))

#         return nn.Sequential(*layers)

#     def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
#         modules = []
#         for i in range(convs):
#             modules.extend([
#                 nn.Conv2d(inplanes, planes, kernel_size=3,
#                           stride=stride if i == 0 else 1,
#                           padding=dilation, bias=False, dilation=dilation),
#                 nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
#                 nn.ReLU(inplace=True)])
#             inplanes = planes
#         return nn.Sequential(*modules)

#     def forward(self, x):
#         y = []
#         x = self.base_layer(x)
#         for i in range(6):
#             x = getattr(self, 'level{}'.format(i))(x)
#             y.append(x)
#         return y

#     def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
#         # fc = self.fc
#         if name.endswith('.pth'):
#             model_weights = torch.load(data + name)
#         else:
#             model_url = get_model_url(data, name, hash)
#             model_weights = model_zoo.load_url(model_url)
#         num_classes = len(model_weights[list(model_weights.keys())[-1]])
#         self.fc = nn.Conv2d(
#             self.channels[-1], num_classes,
#             kernel_size=1, stride=1, padding=0, bias=True)
#         self.load_state_dict(model_weights, strict=False)
#         # self.fc = fc


# def dla34(pretrained=True, **kwargs):  # DLA-34
#     model = DLA([1, 1, 1, 2, 2, 1],
#                 [16, 32, 64, 128, 256, 512],
#                 block=BasicBlock, **kwargs)
#     if pretrained:
#         model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
#     return model

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DeformableConv2d(chi, cho, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class DLASeg_nonlocal(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg_nonlocal, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level

        self.base = coatnet_2()
        # self.base = globals()[base_name](pretrained=pretrained)
        channels = [128, 128, 256, 512, 1026] # self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])

        self.non_local = NLBlockND(channels[self.first_level], dimension=2)

        self.aspp = build_aspp('drn',8,SynchronizedBatchNorm2d)
        self.decoder = build_decoder(256,'drn',SynchronizedBatchNorm2d)


        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(channels[self.first_level], head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)

              if 'tl' or 'bl' or 'br' or 'lm' or 'rm' or 'ct' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)

            else:
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x, low_level_feat = self.base(x)
        # print(x[0].shape)
        # print(x[1].shape)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        # x = self.dla_up(x)

        # y = []
        # for i in range(self.last_level - self.first_level):
        #     print(i)
        #     y.append(x[i].clone())
        # self.ida_up(y, 0, len(y))


        # y_nonlocal = self.non_local(y[-1])
        z = {}
        for head in self.heads:
            # print(head)
            if head == 'ct' or head == 'ct_reg':
                z[head] = self.__getattr__(head)(x)
            else:
                z[head] = self.__getattr__(head)(x)
        return [z]
    

def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
  model = DLASeg_nonlocal('drn_d_{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=8,
                          head_conv=head_conv)
  return model
