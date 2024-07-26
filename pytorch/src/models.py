from model_block import *

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

@torch.jit.script
def arctan_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return epsilon + (1 - 2 * epsilon) * (0.5 + torch.arctan(x)/torch.tensor(np.pi))

@torch.jit.script
def softsign_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return (0.5 - epsilon) * F.softsign(x) + 0.5

@torch.jit.script
def sigmoid_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return torch.sigmoid(x)

@torch.jit.script
def linear_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return epsilon + (1 - 2 * epsilon) * (x - x.min())/(x.max() - x.min())

@torch.jit.script
def inv_square_root_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return (0.5 - epsilon) * x * torch.rsqrt(1 + x ** 2) + 0.5

@torch.jit.script
def cdf_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    # https://github.com/IraKorshunova/pytorch/blob/master/torch/autograd/_functions/pointwise.py#L274
    # https://github.com/IraKorshunova/pytorch/blob/master/torch/lib/THC/THCNumerics.cuh#L441
    # https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g3b8115ff34a107f4608152fd943dbf81
    return (0.5 - epsilon) * torch.erf(x/torch.sqrt(torch.tensor(2))) + 0.5

@torch.jit.script
def hardtanh_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return F.hardtanh(x, epsilon, 1.0 - epsilon)

@torch.jit.script
def no_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.init = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.dropout = nn.Dropout(0.5)
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.dropout2 = nn.Dropout(0.5)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.init(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.dropout(self.down3(x3))
        x5 = self.dropout2(self.down4(x4))
        y1 = self.up1(x5, x4)
        y2 = self.up2(y1, x3)
        y3 = self.up3(y2, x2)
        y4 = self.up4(y3, x1)
        logits = self.outc(y4)
        return logits

    def use_checkpointing(self):
        self.init = torch.utils.checkpoint(self.init)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class Tiny_unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Tiny_unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.init = (DoubleConv(n_channels, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.dropout = nn.Dropout(0.5)
        factor = 2 if bilinear else 1
        self.down3 = (Down(64, 128 // factor))
        self.dropout2 = nn.Dropout(0.5)
        self.up1 = (Up(128, 64 // factor, bilinear))
        self.up2 = (Up(64, 32 // factor, bilinear))
        self.up3 = (Up(32, 16 // factor, bilinear))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x):
        x1 = self.init(x)
        x2 = self.down1(x1)
        x3 = self.dropout(self.down2(x2))
        x4 = self.dropout2(self.down3(x3))
        y1 = self.up1(x4, x3)
        y2 = self.up2(y1, x2)
        y3 = self.up3(y2, x1)
        logits = self.outc(y3)
        return logits

    def use_checkpointing(self):
        self.init = torch.utils.checkpoint(self.init)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class Tiny_unet_v3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Tiny_unet_v3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.init = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        self.dropout = nn.Dropout(0.5)
        factor = 2 if bilinear else 1
        self.down4 = (Down(128, 256 // factor))
        self.dropout2 = nn.Dropout(0.5)
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (UpMod(32, 32, bilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.init(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.dropout(self.down3(x3))
        x5 = self.dropout2(self.down4(x4))
        y1 = self.up1(x5, x4)
        y2 = self.up2(y1, x3)
        y3 = self.up3(y2, x2)
        y4 = self.up4(y3, x1)
        logits = self.outc(y4)
        return logits

    def use_checkpointing(self):
        self.init = torch.utils.checkpoint(self.init)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class MobileUNet(nn.Module):
    """
    Modified UNet with inverted residual block and depthwise seperable convolution
    https://github.com/lafith/Mobile-UNet
    """

    def __init__(self, n_channels, n_classes):
        super(MobileUNet, self).__init__()

        # encoding arm
        self.conv3x3 = self.depthwise_conv(n_channels, 32, p=1, s=2)
        self.irb_bottleneck1 = self.irb_bottleneck(32, 16, 1, 1, 1)
        self.irb_bottleneck2 = self.irb_bottleneck(16, 24, 2, 2, 6)
        self.irb_bottleneck3 = self.irb_bottleneck(24, 32, 3, 2, 6)
        self.irb_bottleneck4 = self.irb_bottleneck(32, 64, 4, 2, 6)
        self.irb_bottleneck5 = self.irb_bottleneck(64, 96, 3, 1, 6)
        self.irb_bottleneck6 = self.irb_bottleneck(96, 160, 3, 2, 6)
        self.irb_bottleneck7 = self.irb_bottleneck(160, 320, 1, 1, 6)
        self.conv1x1_encode = nn.Conv2d(320, 1280, kernel_size=1, stride=1)
        # decoding arm
        self.D_irb1 = self.irb_bottleneck(1280, 96, 1, 2, 6, True)
        self.D_irb2 = self.irb_bottleneck(96, 32, 1, 2, 6, True)
        self.D_irb3 = self.irb_bottleneck(32, 24, 1, 2, 6, True)
        self.D_irb4 = self.irb_bottleneck(24, 16, 1, 2, 6, True)
        self.DConv4x4 = nn.ConvTranspose2d(16, 16, 4, 2, 1, groups=16, bias=False)
        # Final layer: output channel number can be changed as per the usecase
        self.conv1x1_decode = nn.Conv2d(16, n_classes, kernel_size=1, stride=1)

    def depthwise_conv(self, in_c, out_c, k=3, s=1, p=0):
        """
        optimized convolution by combining depthwise convolution and
        pointwise convolution.
        """
        conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=k, padding=p, groups=in_c, stride=s),
            nn.BatchNorm2d(num_features=in_c),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=1),
        )
        return conv

    def irb_bottleneck(self, in_c, out_c, n, s, t, d=False):
        """
        create a series of inverted residual blocks.
        """
        convs = []
        xx = InvertedResidualBlock(in_c, out_c, s, t, deconvolve=d)
        convs.append(xx)
        if n > 1:
            for i in range(1, n):
                xx = InvertedResidualBlock(out_c, out_c, 1, t, deconvolve=d)
                convs.append(xx)
        conv = nn.Sequential(*convs)
        return conv

    def get_count(self, model):
        # simple function to get the count of parameters in a model.
        num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return num

    def forward(self, x):
        # Left arm/ Encoding arm
        # D1
        x1 = self.conv3x3(x)  # (32, 112, 112)
        x2 = self.irb_bottleneck1(x1)  # (16,112,112) s1
        # D2
        x3 = self.irb_bottleneck2(x2)  # (24,56,56) s2
        # D3
        x4 = self.irb_bottleneck3(x3)  # (32,28,28) s3
        # D4
        x5 = self.irb_bottleneck4(x4)  # (64,14,14)
        x6 = self.irb_bottleneck5(x5)  # (96,14,14) s4
        # D5
        x7 = self.irb_bottleneck6(x6)  # (160,7,7)
        x8 = self.irb_bottleneck7(x7)  # (320,7,7)
        # C1
        x9 = self.conv1x1_encode(x8)  # (1280,7,7) s5

        # Right arm / Decoding arm with skip connections
        d1 = self.D_irb1(x9) + x6
        d2 = self.D_irb2(d1) + x4
        d3 = self.D_irb3(d2) + x3
        d4 = self.D_irb4(d3) + x2
        d5 = self.DConv4x4(d4)
        out = self.conv1x1_decode(d5)
        return out

def Lars76_unet(n_channels, n_classes):
    if n_classes <= 3:
        return smp.Unet("resnet34", classes=n_classes, encoder_weights="imagenet", in_channels=n_channels)
    else:
        return smp.Unet("resnet34", classes=n_classes, encoder_weights=None, in_channels=n_channels)


if __name__ == "__main__":
    from torchsummary import summary
    #model = UNet(1,6)
    #model = Tiny_unet_v3(1,6)
    model = MobileUNet(1,6)
    #model = Lars76_unet(1,6)

    #model = smp.Unet("mobilenet_v2", classes=6, in_channels=1)

    print(model)
    summary(model, (1, 256,256))

