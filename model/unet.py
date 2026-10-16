"""
unet.py
=======
Standard UNet Architecture for Liver Disease Segmentation from MRI.
Features:
- Encoder with contracting double convolutions and max pooling
- Bottleneck representation
- Decoder with expansive transposed convolutions and skip connections
- Configurable base features (default: 16 or 32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard UNet for Medical Image Segmentation.
    Encoder-Decoder with Skip Connections.
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=16):
        super(UNet, self).__init__()
        
        f = base_features
        # Contracting Path (Encoder)
        self.inc = DoubleConv(in_channels, f)                 # (B, f, H, W)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(f, f * 2)                              # (B, 2f, H/2, W/2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(f * 2, f * 4)                          # (B, 4f, H/4, W/4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(f * 4, f * 8)                          # (B, 8f, H/8, W/8)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(f * 8, f * 16)                         # (B, 16f, H/16, W/16)
        )

        # Expansive Path (Decoder)
        self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(f * 16, f * 8)

        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(f * 8, f * 4)

        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(f * 4, f * 2)

        self.up4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(f * 2, f)

        # Final 1x1 convolution
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        d1 = self.up1(x5)
        # Pad if odd dimensions
        if d1.shape != x4.shape:
            d1 = F.interpolate(d1, size=x4.shape[2:], mode="bilinear", align_corners=True)
        d1 = torch.cat([x4, d1], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        if d2.shape != x3.shape:
            d2 = F.interpolate(d2, size=x3.shape[2:], mode="bilinear", align_corners=True)
        d2 = torch.cat([x3, d2], dim=1)
        d2 = self.conv_up2(d2)

        d3 = self.up3(d2)
        if d3.shape != x2.shape:
            d3 = F.interpolate(d3, size=x2.shape[2:], mode="bilinear", align_corners=True)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.conv_up3(d3)

        d4 = self.up4(d3)
        if d4.shape != x1.shape:
            d4 = F.interpolate(d4, size=x1.shape[2:], mode="bilinear", align_corners=True)
        d4 = torch.cat([x1, d4], dim=1)
        d4 = self.conv_up4(d4)

        logits = self.outc(d4)
        return logits

    def predict_mask(self, x, threshold=0.5):
        """
        Runs forward pass and applies sigmoid + thresholding to return binary mask.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
        return preds, probs
