import torch.nn as nn
import torch
from torchvision.models import resnet34, resnet50, resnet101, resnet152, resnet18
# from model.backbones.resnet import resnet34, resnet50, resnet101, resnet152,resnet18
from torchsummaryX import summary
import torch.nn.functional as F
from collections import OrderedDict

# from conv_block import Conv
import functools
from functools import partial
import os, sys
from lib.models.networks.pose_dla_dcn import dla34, dla60


__all__ = ['FANet']

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), group=1, bn_act=False,
                 bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)



class FANet(nn.Module):
    def __init__(self, backbone, pretrained=True, classes=11):
        super(FANet, self).__init__()

        if backbone.lower() == "resnet18":
            encoder = resnet18(pretrained=pretrained)
            out_channels = 512
        elif backbone.lower() == "resnet34":
            encoder = resnet34(pretrained=pretrained)
            out_channels = 512
        elif backbone.lower() == "resnet50":
            encoder = resnet50(pretrained=pretrained)
            out_channels = 2048
        elif backbone.lower() == "resnet101":
            encoder = resnet101(pretrained=pretrained)
            out_channels = 2048
        elif backbone.lower() == "resnet152":
            encoder = resnet152(pretrained=pretrained)
        elif backbone.lower() == "dla":
            encoder = dla60(pretrained=True)
            out_channels = 1024
        else:
            raise NotImplementedError("{} Backbone not implemented".format(backbone))


        if backbone.lower() == "dla":
            self.base = encoder.base_layer
            self.maxpool = nn.MaxPool2d(2)
            self.conv2_x = encoder.level1  # 1/4
            self.conv3_x = encoder.level2  # 1/8
            self.conv4_x = encoder.level3  # 1/16
            self.conv5_x = encoder.level4  # 1/32
            self.conv6_x = encoder.level5  # 1/32

        else:
            self.conv = encoder.conv1
            self.bn1 = encoder.bn1
            self.relu = encoder.relu
            self.maxpool = encoder.maxpool
            self.conv2_x = encoder.layer1  # 1/4
            self.conv3_x = encoder.layer2  # 1/8
            self.conv4_x = encoder.layer3  # 1/16
            self.conv5_x = encoder.layer4  # 1/32
            
        if backbone.lower() == "dla":
            self.gr = GR(1024, 1024,4) 
            channels = [64, 128, 256, 512, 512]
            self.channels = channels
            self.lr1 = LR(channels[1])
            self.lr2 = LR(channels[2])
            self.lr3 = LR(channels[3])
            self.lr4 = LR(channels[4])
            
            self.a2 = A(128, 64)
            self.a3 = A(256, 64)
            self.a4 = A(512, 64)
            self.a5 = A(512, 64)

            
            self.fuse = conv_block(320, 256, 3, 1, padding=1, bn_act=True)
            
            self.d5 = DModule(1024, 512)
            self.d4 = DModule(512, 256)
            self.d3 = DModule(256, 128)
            self.d2 = DModule(128, 64)

            self.classifier = Classifier(64, classes)



        else:
            self.gr = GR(512, 512,4)       
            channels = [64, 64, 128, 256, 512]
            self.channels = channels
            self.lr1 = LR(channels[1])
            self.lr2 = LR(channels[2])
            self.lr3 = LR(channels[3])
            self.lr4 = LR(channels[4])
            
            self.a2 = A(64, 64)
            self.a3 = A(128, 64)
            self.a4 = A(256, 64)
            self.a5 = A(512, 64)

            self.fuse = conv_block(320, 128, 3, 1, padding=1, bn_act=True)

            self.d5 = DModule(512, 256)
            self.d4 = DModule(256, 128)
            self.d3 = DModule(128, 64)
            self.d2 = DModule(64, 64)

            self.classifier = Classifier(64, classes)

    def forward(self, x):
        B, C, H, W = x.size()
        if self.base is not None:
            x = self.base(x)
            x = self.maxpool(x)

            x2 = self.conv2_x(x)
            x3 = self.conv3_x(x2)
            x4 = self.conv4_x(x3)
            x5 = self.conv5_x(x4)
            x6 = self.conv6_x(x5)
            g = self.gr(x6)

            d5 = self.d5(x6,g)  
            d4 = self.d4(x5,d5)
            d3 = self.d3(x4,d4)
            d2 = self.d2(x3,d3)

            
            g1 = self.lr1(x3) 
            g2 = self.lr2(x4) 
            g3 = self.lr3(x5)  
            g4 = self.lr4(x5) 

            o2 = self.a2(g1)
            o3 = self.a3(g2)
            o4 = self.a4(g3)
            o5 = self.a5(g4)

            o2 = F.interpolate(o2, size=x2.size()[2:], mode="bilinear")
            o3 = F.interpolate(o3, size=x2.size()[2:], mode="bilinear")
            o4 = F.interpolate(o4, size=x2.size()[2:], mode="bilinear")
            o5 = F.interpolate(o5, size=x2.size()[2:], mode="bilinear")
            d2 = F.interpolate(d2, size=x2.size()[2:], mode="bilinear")
      

            final = torch.cat((o2,d2,o3,o4,o5), dim=1)
            final = self.fuse(final)

            
            return final


        else:
            x = self.conv(x)
            x = self.bn1(x)
            x1 = self.relu(x)

            x = self.maxpool(x1)

            x2 = self.conv2_x(x)
            x3 = self.conv3_x(x2)
            x4 = self.conv4_x(x3)
            x5 = self.conv5_x(x4)
            # print(x5.shape)
            g = self.gr(x5)

            d5 = self.d5(x5,g)  
            d4 = self.d4(x4,d5)
            d3 = self.d3(x3,d4)
            d2 = self.d2(x2,d3)

            
            g1 = self.lr1(x2) 
            g2 = self.lr2(x3) 
            g3 = self.lr3(x4)  
            g4 = self.lr4(x5) 

            o2 = self.a2(g1)
            o3 = self.a3(g2)
            o4 = self.a4(g3)
            o5 = self.a5(g4)

            o2 = F.interpolate(o2, size=x2.size()[2:], mode="bilinear")
            o3 = F.interpolate(o3, size=x2.size()[2:], mode="bilinear")
            o4 = F.interpolate(o4, size=x2.size()[2:], mode="bilinear")
            o5 = F.interpolate(o5, size=x2.size()[2:], mode="bilinear")
            d2 = F.interpolate(d2, size=x2.size()[2:], mode="bilinear")
        

            final = torch.cat((o2,d2,o3,o4,o5), dim=1)
            final = self.fuse(final)

            
            return final


class Module(nn.Module):
    def __init__(self, in_channels: int, groups=2) -> None:
        super(Module, self).__init__()
        self.groups = groups
        self.conv_dws1 = nn.Sequential(
            conv_block(in_channels, 4*in_channels, kernel_size=3, stride=1, padding=4,
                                    group=1, dilation=4, bn_act=True),
            nn.PixelShuffle(upscale_factor=2))
        self.conv_dws2 = nn.Sequential(
            conv_block(in_channels, 4*in_channels, kernel_size=3, stride=1, padding=8,
                                    group=1, dilation=8, bn_act=True),
            nn.PixelShuffle(upscale_factor=2))

        self.fusion = nn.Sequential(
            conv_block(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True))

        self.conv_dws3 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        br1 = self.conv_dws1(x)
        b2 = self.conv_dws1(x)

        out = torch.cat((br1, b2), dim=1)
        out = self.fusion(out)

        br3 = self.conv_dws3(F.adaptive_avg_pool2d(x, (1, 1)))
        output = br3 + out

        return output


class GR(nn.Module):
    def __init__(self, in_channels, out_channels, num_splits):
        super(GR,self).__init__()

        assert in_channels % num_splits == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_splits = num_splits
        self.subspaces = nn.ModuleList(
            [Module(int(self.in_channels / self.num_splits)) for i in range(self.num_splits)])

        self.out = conv_block(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)

    def forward(self, x):
        group_size = int(self.in_channels / self.num_splits)
        sub_Feat = torch.chunk(x, self.num_splits, dim=1)
        out = []
        for id, l in enumerate(self.subspaces):
            out.append(self.subspaces[id](sub_Feat[id]))
        out = torch.cat(out, dim=1)
        out = self.out(out)
        return out


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)

    return x


class LR(nn.Module):
    def __init__(self, in_channels: int, groups=2) -> None:
        super(LR, self).__init__()
        self.groups = groups
        self.conv_dws1 = conv_block(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0,
                                    group=in_channels // 2, bn_act=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pw1 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.softmax = nn.Softmax(dim=1)

        self.conv_dws2 = conv_block(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0,
                                    group=in_channels // 2,
                                    bn_act=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pw2 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=False)

        self.branch3 = nn.Sequential(
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, group=in_channels, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True)) 

    def forward(self, x):
        x0, x1 = x.chunk(2, dim=1)
        out1 = self.conv_dws1(x0)
        out1 = self.maxpool1(out1)
        out1 = self.conv_pw1(out1)

        out2 = self.conv_dws1(x1)
        out2 = self.maxpool1(out2)
        out2 = self.conv_pw1(out2)

        out = torch.add(out1, out2)

        b, c, h, w = out.size()
        out = self.softmax(out.view(b, c, -1))
        out = out.view(b, c, h, w)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)
        out = torch.add(out, x)
        out = channel_shuffle(out, groups=self.groups)

        br3 = self.branch3(x)

        output = br3 + out

        return output



class DModule(nn.Module):
    def __init__(self, in_channels, out_channels, red=1):
        super(DModule, self).__init__()
     
        self.conv1 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.conv2 = conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bn_act=True)
        self.conv3 = nn.Sequential(
            conv_block(2 * in_channels, 4 * out_channels, kernel_size=3, stride=1, padding=1, bn_act=True),
            nn.PixelShuffle(upscale_factor=2))

    def forward(self, x_g, y_high):
        h, w = x_g.size(2), x_g.size(3)

        y_high = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y_high)
        x_g = self.conv1(x_g)
        y_high = self.conv2(y_high)
       
        out = torch.cat([y_high, x_g], 1)

        out = self.conv3(out)

        return out


class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Classifier, self).__init__()
        self.fc = conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.fc(x)


class A(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(A, self).__init__()
        self.conv = conv_block(in_channels, out_channels, 1, 1, padding=0, bn_act=True)

    def forward(self, x):
        return self.conv(x)




if __name__ == "__main__":
    input_tensor = torch.rand(2, 3, 512, 1024)
    model = SPFNet("resnet18")
    summary(model, input_tensor)




