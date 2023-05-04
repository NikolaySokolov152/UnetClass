"""
Parts of the U-Net model
copy from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpMod(nn.Module):
    """
        my version
        Upscaling then double conv
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            if in_channels > out_channels:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels*2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

"""
inverted residual block used in MobileNetV2
copy from https://github.com/lafith/Mobile-UNet/blob/main/model.py
article Mobile-Unet: An efficient convolutional neural network for fabric defect detection
"""
class InvertedResidualBlock(nn.Module):
    """
    inverted residual block used in MobileNetV2
    """

    def __init__(self, in_c, out_c, stride, expansion_factor=6, deconvolve=False):
        super(InvertedResidualBlock, self).__init__()
        # check stride value
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor or t as mentioned in the paper
        ex_c = int(self.in_c * expansion_factor)
        if deconvolve:
            self.conv = nn.Sequential(
                # pointwise convolution
                nn.Conv2d(self.in_c, ex_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ex_c),
                nn.ReLU6(inplace=True),
                # depthwise convolution
                nn.ConvTranspose2d(ex_c, ex_c, 4, self.stride, 1, groups=ex_c, bias=False),
                nn.BatchNorm2d(ex_c),
                nn.ReLU6(inplace=True),
                # pointwise convolution
                nn.Conv2d(ex_c, self.out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.out_c),
            )
        else:
            self.conv = nn.Sequential(
                # pointwise convolution
                nn.Conv2d(self.in_c, ex_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ex_c),
                nn.ReLU6(inplace=True),
                # depthwise convolution
                nn.Conv2d(ex_c, ex_c, 3, self.stride, 1, groups=ex_c, bias=False),
                nn.BatchNorm2d(ex_c),
                nn.ReLU6(inplace=True),
                # pointwise convolution
                nn.Conv2d(ex_c, self.out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.out_c),
            )
        self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

    def forward(self, x):
        if self.use_skip_connection:
            out = self.conv(x)
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        else:
            return self.conv(x)