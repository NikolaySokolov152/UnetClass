import torch
import os

import torch.nn as nn
from torch.nn import functional as F
import numpy as np


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

class Tiny_unet_v3(nn.Module):
    def __name__(self):
        return("Tiny_unet_v3")

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

model_path = "Multiple_diffusion_42_slices_6_classes/model_by_config_diffusion_data_42_slices_6_classes_dataset_mix_5_classes_seed_1253865913_tiny_unet_v3.pth"

model = Tiny_unet_v3(1,5)
model.load_state_dict(torch.load(model_path))
print(model)


model.to("cpu")

import cv2
import numpy as np

@torch.jit.script
def sigmoid_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return torch.sigmoid(x)

img = cv2.imread("data/original data/testing/original/testing0000.png", 0)

cv2.imshow("input", img.copy())
cv2.waitKey()

#img = cv2.resize(img, (256,256))
img = img.astype(float) / 255

input_img = np.reshape(img, (1,) + img.shape + (1,))

torch_img = torch.from_numpy(np.array(input_img)).type(torch.FloatTensor).permute(0, 3, 1, 2)

print(torch_img.shape)

outputs = model(torch_img)
outputs = sigmoid_activation(outputs, 1e-7)

result = outputs.detach().permute(0, 2, 3, 1).numpy()[0]

masks = (result*255).astype(np.uint8)
for i in range(5):
    cv2.imshow(f"predict {i} class", masks[:,:,i])
cv2.waitKey()
