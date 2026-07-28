import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class AttentionGate(nn.Module):
    """
    Attention Gate to focus on thin solar filament features in skip connections.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_l(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class SolarFilamentAttentionUNet(nn.Module):
    """
    High-capacity 5-Level Deep Attention U-Net tailored for fine solar filament segmentation.
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=32):
        super(SolarFilamentAttentionUNet, self).__init__()
        
        f = [base_features, base_features*2, base_features*4, base_features*8, base_features*16]
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, f[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = ConvBlock(f[0], f[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = ConvBlock(f[1], f[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = ConvBlock(f[2], f[3])
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(f[3], f[4])
        
        # Attention Gates
        self.att4 = AttentionGate(F_g=f[3], F_l=f[3], F_int=f[2])
        self.att3 = AttentionGate(F_g=f[2], F_l=f[2], F_int=f[1])
        self.att2 = AttentionGate(F_g=f[1], F_l=f[1], F_int=f[0])
        self.att1 = AttentionGate(F_g=f[0], F_l=f[0], F_int=f[0]//2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f[4], f[3])
        
        self.up3 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f[3], f[2])
        
        self.up2 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f[2], f[1])
        
        self.up1 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f[1], f[0])
        
        # Final Classification Head
        self.final_conv = nn.Conv2d(f[0], out_channels, kernel_size=1)

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
        
        # Decoder with Attention Skip Connections
        d4 = self.up4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)
        
        logits = self.final_conv(d1)
        return logits

# Alias for backward compatibility
SolarFilamentUNet = SolarFilamentAttentionUNet

if __name__ == "__main__":
    model = SolarFilamentAttentionUNet()
    x = torch.randn(2, 1, 512, 512)
    out = model(x)
    print(f"Model forward pass success! Input: {x.shape}, Output logits: {out.shape}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
