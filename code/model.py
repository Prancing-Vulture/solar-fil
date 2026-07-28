# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

class ResConvBlock(nn.Module):
    """
    Residual Convolutional Block with BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(ResConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x), inplace=True)

class SolarFilamentUNet(nn.Module):
    """
    U-Net architecture tailored for Solar Filament Segmentation.
    Accepts 1-channel solar images and returns raw logits.
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=32):
        super(SolarFilamentUNet, self).__init__()
        
        # Encoder (Contracting Path)
        self.enc1 = ResConvBlock(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = ResConvBlock(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = ResConvBlock(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = ResConvBlock(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = ResConvBlock(base_features * 8, base_features * 16)
        
        # Decoder (Expanding Path with Skip Connections)
        self.up4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.dec4 = ResConvBlock(base_features * 16, base_features * 8)
        
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = ResConvBlock(base_features * 8, base_features * 4)
        
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = ResConvBlock(base_features * 4, base_features * 2)
        
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = ResConvBlock(base_features * 2, base_features)
        
        # Final Output Layer (returns raw logits)
        self.final_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder with Skip Connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        logits = self.final_conv(d1)
        return logits

if __name__ == "__main__":
    model = SolarFilamentUNet()
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    print(f"Model instantiated successfully! Input shape: {x.shape}, Output logits shape: {out.shape}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
