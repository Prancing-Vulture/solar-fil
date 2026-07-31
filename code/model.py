import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SolarFilamentAttentionUNet(nn.Module):
    """
    Wrapper for Segmentation Models PyTorch (SMP) U-Net architecture.
    Uses pretrained ResNet-34 backbone by default.
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=32):
        super(SolarFilamentAttentionUNet, self).__init__()
        # We ignore base_features since we are using a fixed pretrained backbone (resnet34)
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels
        )

    def forward(self, x):
        return self.model(x)

# Alias for backward compatibility
SolarFilamentUNet = SolarFilamentAttentionUNet

if __name__ == "__main__":
    model = SolarFilamentAttentionUNet()
    x = torch.randn(2, 1, 512, 512)
    out = model(x)
    print(f"Model forward pass success! Input: {x.shape}, Output logits: {out.shape}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
