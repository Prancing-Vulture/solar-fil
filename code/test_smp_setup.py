import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

def main():
    print("Testing SMP model instantiation...")
    # Instantiate a Unet model with resnet34 backbone
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1, # grayscale input
        classes=1 # binary output (1 channel)
    )
    print("Model created successfully!")
    
    # Run a test forward pass
    dummy_input = torch.randn(2, 1, 512, 512)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test albumentations pipeline
    print("Testing Albumentations pipeline...")
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.2),
        ToTensorV2()
    ])
    
    dummy_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    
    augmented = transform(image=dummy_image, mask=dummy_mask)
    img_tensor = augmented['image']
    mask_tensor = augmented['mask']
    
    print(f"Augmented image tensor shape: {img_tensor.shape}, type: {img_tensor.dtype}")
    print(f"Augmented mask tensor shape: {mask_tensor.shape}, type: {mask_tensor.dtype}")
    print("All checks passed successfully!")

if __name__ == "__main__":
    main()
