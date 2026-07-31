import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.dirname(__file__))

from analysis import path as BASE_DATASET_PATH
from model import SolarFilamentUNet

def run_inference(checkpoint_path="checkpoints/best_model.pth", num_samples=5, img_size=(512, 512), output_dir="inference_results"):
    print("=" * 60)
    print("SOLAR FILAMENT INFERENCE & TEST EVALUATION")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model architecture & trained weights
    model = SolarFilamentUNet(in_channels=1, out_channels=1, base_features=32).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model weights from Epoch {checkpoint.get('epoch', 'N/A')} (Best Val Dice: {checkpoint.get('val_dice', 0.0):.4f})")
    
    # Locate test images
    magfilo_dir = os.path.join(BASE_DATASET_PATH, "MAGFiLO_1.0_Kaggle_2026")
    test_img_dir = os.path.join(magfilo_dir, "test", "test_images")
    
    if not os.path.exists(test_img_dir):
        print(f"ERROR: Test image directory not found at {test_img_dir}")
        return
        
    test_files = sorted(os.listdir(test_img_dir))[:num_samples]
    print(f"Processing {len(test_files)} test images for segmentation visualization...")
    
    fig, axes = plt.subplots(len(test_files), 3, figsize=(12, 4 * len(test_files)))
    if len(test_files) == 1:
        axes = np.expand_dims(axes, 0)
        
    with torch.no_grad():
        for idx, fname in enumerate(test_files):
            img_path = os.path.join(test_img_dir, fname)
            pil_img = Image.open(img_path).convert('L')
            orig_size = pil_img.size
            
            pil_img_resized = pil_img.resize(img_size, Image.BILINEAR)
            img_np_uint8 = np.array(pil_img_resized, dtype=np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_np_clahe = clahe.apply(img_np_uint8)
            img_np = img_np_clahe.astype(np.float32) / 255.0
            
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
            
            # Forward pass
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_mask = (probs > 0.5).astype(np.float32)
            
            # Render visualization row
            # 1. Original Input Image
            axes[idx, 0].imshow(img_np, cmap='gray')
            axes[idx, 0].set_title(f"Test Image: {fname}", fontsize=10)
            axes[idx, 0].axis('off')
            
            # 2. Predicted Probability Heatmap
            axes[idx, 1].imshow(probs, cmap='magma')
            axes[idx, 1].set_title("Predicted Sigmoid Probability", fontsize=10)
            axes[idx, 1].axis('off')
            
            # 3. Green Filament Overlay
            img_normalized = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
            overlay = np.stack([img_normalized]*3, axis=-1)
            overlay[pred_mask > 0] = [0.1, 0.95, 0.2]
            
            axes[idx, 2].imshow(overlay)
            axes[idx, 2].set_title("Predicted Filament Segmentation", fontsize=10)
            axes[idx, 2].axis('off')
            
    plt.tight_layout()
    output_plot_path = os.path.join(output_dir, "test_segmentation_predictions.png")
    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    
    print(f"\nInference finished successfully! Results saved to: {output_plot_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_inference()
