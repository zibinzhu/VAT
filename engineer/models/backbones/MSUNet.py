import torch
import torch.nn as nn
import torch.nn.functional as F
from engineer.models.registry import BACKBONES
from .base_backbone import _BaseBackbone

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_activate=True):
        super().__init__()
        self.is_activate = is_activate
        self.conv_norm = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(32, out_channels),
        )

    def forward(self, x):
        x = self.conv_norm(x)
        if self.is_activate:
            x = F.leaky_relu(x, negative_slope=0.1, inplace=True)
        return x


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


class EnConv(nn.Module):
    def __init__(self, l_channels, h_channels, pool_kernel_size=2):
        super().__init__()
        
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        # self.avg_pool = nn.MaxPool2d(pool_kernel_size)
        self.conv1 = ConvNorm(l_channels+h_channels, l_channels)
        self.max_pool = nn.MaxPool2d(2)
        # self.max_pool = nn.AvgPool2d(2)
        self.conv2 = ConvNorm(l_channels, l_channels, is_activate=False)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x_l, x_h):
        x_h = self.avg_pool(x_h)
        x_l = self.conv1(torch.cat([x_h, x_l], dim=1))

        res = x_l 
        x_l = self.max_pool(x_l)
        x_l = self.conv2(x_l)
        x_l = self.up(x_l)

        return F.leaky_relu(x_l+res, negative_slope=0.1, inplace=True)

class DeConv(nn.Module):
    def __init__(self, l_channels, h_channels, scale_factor=2):
        super().__init__()
        
        self.up_conv = ConvNorm(l_channels, h_channels, kernel_size=1, stride=1, padding=0, is_activate=False)
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

    def forward(self, x_l, x_h):
        x_l = self.up_conv(x_l)
        x_l = self.up(x_l)
        return F.leaky_relu(x_h+x_l, negative_slope=0.1, inplace=True)


class Filter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.enconv1 = EnConv(channels, channels//2, pool_kernel_size=2)
        self.enconv2 = EnConv(channels, channels//4, pool_kernel_size=4)
        self.enconv3 = EnConv(channels, channels//8, pool_kernel_size=8)

        self.deconv1 = DeConv(channels, channels//2, scale_factor=2)
        self.deconv2 = DeConv(channels, channels//4, scale_factor=4)
        self.deconv3 = DeConv(channels, channels//8, scale_factor=8)

        # self.final_conv = ConvNorm(channels, channels)
        self.final_conv = DoubleConv(channels*2, channels)

    def forward(self, x, x3, x2, x1):
        x4 = x
        x = self.enconv1(x, x3)
        x3 = self.deconv1(x, x3)
        x = self.enconv2(x, x2)
        x2 = self.deconv2(x, x2)
        x = self.enconv3(x, x1)
        x1 = self.deconv3(x, x1)
        #x = self.final_conv(x)
        x = self.final_conv(torch.cat([x4, x], dim=1))
        return x, x3, x2, x1


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, is_down=True, is_pool=True):
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
class MSUNet(_BaseBackbone):
    def __init__(self, in_channels, hidden_channels):
        super(MSUNet, self).__init__()
        self.name = 'MSUNet Backbone'
        self.input_para={'in_channels':in_channels,'hidden_channels':hidden_channels}

        self.down = Down(in_channels, hidden_channels, is_down=False)
        self.down1 = Down(hidden_channels, hidden_channels*2, is_pool=True)
        self.down2 = Down(hidden_channels*2, hidden_channels*4, is_pool=True)
        self.down3 = Down(hidden_channels*4, hidden_channels*8, is_pool=True)
        
        self.filter = Filter(hidden_channels*8)

        self.up1 = Up(hidden_channels*8, hidden_channels*4)
        self.up2 = Up(hidden_channels*4, hidden_channels*2)
        self.up3 = Up(hidden_channels*2, hidden_channels)

    def forward(self, x1):
        x1 = self.down(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4, x3, x2, x1 = self.filter(x4, x3, x2, x1)

        x = self.up1(x4, x3)
        x_c = x
        x = self.up2(x, x2)
        x_s = x
        x = self.up3(x, x1)

        return x_c, x_s, x