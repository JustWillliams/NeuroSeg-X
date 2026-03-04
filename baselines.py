import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.out_seg = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        d1 = torch.cat([self.up1(e4), e3], dim=1)
        d1 = self.dec1(d1)
        d2 = torch.cat([self.up2(d1), e2], dim=1)
        d2 = self.dec2(d2)
        d3 = torch.cat([self.up3(d2), e1], dim=1)
        d3 = self.dec3(d3)
        
        seg = self.out_seg(d3)
        return seg

# Placeholder for TransUNet, Swin-UNet, BiTr-UNet (Simplified for implementation)
class TransUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()
        self.unet = UNet(in_channels, out_channels) # Using UNet as proxy base for now
    def forward(self, x): return self.unet(x)

class SwinUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()
        self.unet = UNet(in_channels, out_channels)
    def forward(self, x): return self.unet(x)

class BiTrUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()
        self.unet = UNet(in_channels, out_channels)
    def forward(self, x): return self.unet(x)
