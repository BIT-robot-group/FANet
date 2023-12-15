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

from .agatt import AggAtt
from .DCNv2.dcn_v2 import DCN


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, ws=False):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.ws = ws

    def forward(self, input):
        weight = self.weight
        if self.ws:
            weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                      keepdim=True).mean(dim=3, keepdim=True)
            weight = weight - weight_mean
            std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
            weight = weight / std.expand_as(weight)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):

        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        att_y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y, att_y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights, strict=False)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

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



class Conv3x3(nn.Module):
    def __init__(self, chi, cho):
        super(Conv3x3, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(chi, cho, kernel_size=3, stride=1,
                     padding=1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x

class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1,
                        padding=1, dilation=1, deformable_groups=1)

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
        #getattr(self, 'node_' + str(len(channels)-1)).register_backward_hook(ig)


    def forward(self, layers, startp, endp):
        #offset_nodes = []
        #offset_projs = []
        out_layers = []
        for i in range(startp+1):
            out_layers.append(layers[i])
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            cur_layer = project(layers[i])
            cur_layer = upsample(cur_layer)
            node = getattr(self, 'node_' + str(i - startp))
            cur_layer = node(cur_layer + out_layers[i - 1])
            out_layers.append(cur_layer)
        return out_layers


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

    def forward(self, layers, seg_map=None):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            layers = ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out
        #return out, offset_node_list, offset_proj_list


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class SegHead(nn.Module):

    def __init__(self, in_channel):
        super(SegHead, self).__init__()
        self.conv1 = Conv3x3(in_channel, 256)
        self.conv2 = Conv3x3(256, 256)
        self.pred = nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=1, bias=True)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pred(x)
        x = self.softmax(x)
        return x


class LargeConvHM(nn.Module):

    def __init__(self, num_input, num_hidden, num_output, num_span):
        super(LargeConvHM, self).__init__()
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.num_span = num_span

        self.convh = nn.Sequential(
            nn.Conv2d(num_input, num_hidden,
                      stride=1, kernel_size=(num_span, 1), padding=(num_span//2,0)),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_output,
                      stride=1, kernel_size=(1, num_span), padding=(0,num_span//2)),
        )

        self.convw = nn.Sequential(
            nn.Conv2d(num_input, num_hidden,
                      stride=1, kernel_size=(1, num_span), padding=(0,num_span//2)),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_output,
                      stride=1, kernel_size=(num_span, 1), padding=(num_span//2,0))
        )

    def forward(self, input):
        return self.convh(input) + self.convw(input)


class DLASeg(nn.Module):

    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0, opt=None):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        self.opt = opt
        self.num_class = 18

        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)


        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                nn.Conv2d(channels[self.first_level], head_conv,
                  stride=1, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, classes,
                  kernel_size=final_kernel, stride=1,
                  padding=final_kernel // 2, bias=True))
              if 'hm' == head or 'cor_att' == head:
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
              if 'hm' in head or 'att' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

        # if opt.use_agg_att:
        input_channels = channels[self.first_level]
        final_channels = 18
        # if self.opt.use_agg_att_ctreg:
        #     final_channels += 2
        self.agg_att_lm = AggAtt(input_channels, head_conv,
                            stride=1, kernel_size=3, padding=1,
                            final_channels=final_channels)
        self.agg_att_rm = AggAtt(input_channels, head_conv,
                            stride=1, kernel_size=3, padding=1,
                            final_channels=final_channels)
        self.agg_att_ct = AggAtt(input_channels, head_conv,
                            stride=1, kernel_size=3, padding=1,
                            final_channels=final_channels)


class DLAFPN(nn.Module):
    def __init__(self, base_name, heads, head_conv=128,
                 num_filters=[256, 256, 256],
                 gn=False, ws=False, freeze_bn=False,down_ratio=4):
        super().__init__()

        self.heads = heads
        self.num_class = 36
        self.base = globals()[base_name]()
        channels = self.base.channels
        num_bottleneck_filters = 512
        self.first_level = int(np.log2(down_ratio))
        input_channels = channels[self.first_level]
        final_channels = 36

        if freeze_bn:
            for m in self.base.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        self.lateral4 = nn.Sequential(
            Conv2d(num_bottleneck_filters, num_filters[0],
                   kernel_size=1, bias=False, ws=ws),
            nn.GroupNorm(32, num_filters) if gn else nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True))
        self.lateral3 = nn.Sequential(
            Conv2d(num_bottleneck_filters // 2, num_filters[0],
                   kernel_size=1, bias=False, ws=ws),
            nn.GroupNorm(32, num_filters[0]) if gn else nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True))
        self.lateral2 = nn.Sequential(
            Conv2d(num_bottleneck_filters // 4, num_filters[1],
                   kernel_size=1, bias=False, ws=ws),
            nn.GroupNorm(32, num_filters[1]) if gn else nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True))
        self.lateral1 = nn.Sequential(
            Conv2d(num_bottleneck_filters // 8, num_filters[2],
                   kernel_size=1, bias=False, ws=ws),
            nn.GroupNorm(32, num_filters) if gn else nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True))

        self.decode3 = nn.Sequential(
            Conv2d(num_filters[0], num_filters[1],
                   kernel_size=3, padding=1, bias=False, ws=ws),
            nn.GroupNorm(32, num_filters[1]) if gn else nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True))
        self.decode2 = nn.Sequential(
            Conv2d(num_filters[1], num_filters[2],
                   kernel_size=3, padding=1, bias=False, ws=ws),
            nn.GroupNorm(32, num_filters[2]) if gn else nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True))
        self.decode1 = nn.Sequential(
            Conv2d(num_filters[2], num_filters[2],
                   kernel_size=3, padding=1, bias=False, ws=ws),
            nn.GroupNorm(32, num_filters[2]) if gn else nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True))

        
        self.agg_att_lm = AggAtt(input_channels, head_conv,
                            stride=1, kernel_size=3, padding=1,
                            final_channels=final_channels)
        self.agg_att_rm = AggAtt(input_channels, head_conv,
                            stride=1, kernel_size=3, padding=1,
                            final_channels=final_channels)
        self.agg_att_ct = AggAtt(input_channels, head_conv,
                            stride=1, kernel_size=3, padding=1,
                            final_channels=final_channels)

        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                Conv2d(num_filters[2], head_conv,
                       kernel_size=3, padding=1, bias=False, ws=ws),
                nn.GroupNorm(32, head_conv) if gn else nn.BatchNorm2d(head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output,
                          kernel_size=1))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            if 'tl' or 'bl' or 'br' or 'lm' or 'rm' or 'ct' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def forward(self, x):
        # x, att_x = self.base(x)
        # shapes = [(f.shape[2], f.shape[3]) for f in x]

        # z = {}

        # x = self.dla_up(x)


        # y = []
        # for i in range(self.last_level - self.first_level):
        #     y.append(x[i].clone())
        # y = self.ida_up(y, 0, len(y))

        x, att_x = self.base(x)

        lat4 = self.lateral4(x[-1])
        lat3 = self.lateral3(x[-2])
        lat2 = self.lateral2(x[-3])
        lat1 = self.lateral1(x[-4])

        map4 = lat4
        map3 = lat3 + F.interpolate(map4, scale_factor=2, mode="nearest")
        map3 = self.decode3(map3)
        map2 = lat2 + F.interpolate(map3, scale_factor=2, mode="nearest")
        map2 = self.decode2(map2)
        map1 = lat1 + F.interpolate(map2, scale_factor=2, mode="nearest")
        map1 = self.decode1(map1)

        z = {}
        
        for head in self.heads:
            z[head] = self.__getattr__(head)(map1)
        
        z['ct'] = z['ct'].repeat(1,self.num_class,1,1)
        # if self.opt.use_agg_att:
        # print(z['ct'].shape)
        # print(z['lm'].shape)
        feat = map1
        
        agg_att_lm = self.agg_att_lm(feat, z['lm'])
        agg_att_rm = self.agg_att_rm(feat, z['rm'])
        agg_att_ct = self.agg_att_ct(feat, z['ct'])
        # print(agg_att_lm.shape)
        # print(z['lm'].detach().shape)
        n = self.num_class
        z['lm'] = agg_att_lm[:,:n] + z['lm'].detach()
        z['rm'] = agg_att_rm[:,:n] + z['lm'].detach()
        z['ct'] = agg_att_rm[:,:n] + z['ct'].detach()
        z['ct'] = torch.mean(z['ct'], dim = 1, keepdim=True)
        # print(z['ct'].shape)
        # print(z['lm'].shape)
        # if self.opt.use_agg_att_ctreg:
        #     z['final_reg'] = agg_att[:,2:4] + z['reg'].detach()

        return [z]
    

# def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4, opt=None):

#   model = DLAFPN('dla{}'.format(num_layers), heads,
#                  pretrained=True,
#                  down_ratio=down_ratio,
#                  final_kernel=1,
#                  last_level=5,
#                  head_conv=head_conv,
#                  opt=opt)
#   return model

def get_pose_net(num_layers, heads, head_conv=256,
              num_filters=[256, 256, 256],
              gn=False, ws=False, freeze_bn=False):
    model = DLAFPN('dla34', heads, head_conv=head_conv,
                   num_filters=num_filters,
                   gn=gn, ws=ws, freeze_bn=freeze_bn,down_ratio=16)
    # state_dict_old = torch.load('pretrained_weights/%s.pth' %pretrained)['state_dict']
    # state_dict = OrderedDict()
    # for key, val in state_dict_old.items():
    #     if 'hm' in key or 'wh' in key or 'reg' in key:
    #         continue
    #     state_dict['.'.join(key.split('.')[1:])] = val
    # model.load_state_dict(state_dict, strict=False)

    return model