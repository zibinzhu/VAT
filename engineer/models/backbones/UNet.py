import torch
import torch.nn as nn
from engineer.models.registry import BACKBONES
from .base_backbone import _BaseBackbone
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
       
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, is_down=True, is_pool=True, fine_pifu=False):
        super().__init__()

        if is_down:
            if is_pool:
                self.down = nn.Sequential(
                    nn.MaxPool2d(2),
                    DoubleConv(in_channels, out_channels)
                )
            else:
                self.down = DoubleConv(in_channels, out_channels, stride=2)
        else:
            if fine_pifu:
                self.down =  DoubleConv(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
            else:
                self.down =  DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.GroupNorm(32, out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.up_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.up_conv(x)


@BACKBONES.register_module
class UNet(_BaseBackbone):
    def __init__(self, in_channels, hidden_channels, out_channels=3, fine_pifu=False):
        super(UNet, self).__init__()
        self.name = 'UNet Backbone'
        self.input_para={'in_channels':in_channels,'hidden_channels':hidden_channels}

        self.inc = Down(in_channels, hidden_channels, is_down=False, fine_pifu=fine_pifu)
        self.down1 = Down(hidden_channels, hidden_channels*2, is_pool=True)
        self.down2 = Down(hidden_channels*2, hidden_channels*4, is_pool=True)
        self.down3 = Down(hidden_channels*4, hidden_channels*8, is_pool=True)
        self.down4 = Down(hidden_channels*8, hidden_channels*16, is_pool=True)

        self.up1 = Up(hidden_channels*16, hidden_channels*8)
        self.up2 = Up(hidden_channels*8, hidden_channels*4)
        self.up3 = Up(hidden_channels*4, hidden_channels*2)
        self.up4 = Up(hidden_channels*2, hidden_channels)

        # self.last = nn.Conv2d(hidden_channels, 3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4 = self.up1(x5, x4)
        x3 = self.up2(x4, x3)
        x2 = self.up3(x3, x2)
        x1 = self.up4(x2, x1)

        # n = self.last(x1)

        return x3, x2, x1
