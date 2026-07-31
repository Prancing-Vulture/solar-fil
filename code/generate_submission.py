import os
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.append(os.path.dirname(__file__))

from analysis import path as BASE_DATASET_PATH
from model import SolarFilamentAttentionUNet

def mask_to_coco_rle(binary_mask):
    """
    Encodes a binary numpy mask (H, W) into COCO compressed RLE ASCII string.
    """
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle['counts']

def generate_submission(
    checkpoint_path="checkpoints/best_model.pth",
    output_dir="submission_output",
    img_size=(512, 512),
    orig_size=(2048, 2048),
    min_area_pixels=300,
    prob_threshold=0.4
):
    print("=" * 75)
    print("SOLAR FILAMENT SEGMENTATION - KAGGLE SUBMISSION GENERATOR")
    print("=" * 75)
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
    model = SolarFilamentAttentionUNet(in_channels=1, out_channels=1, base_features=32).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Successfully loaded trained weights from Epoch {checkpoint.get('epoch', 'N/A')} (Best Val Dice: {checkpoint.get('val_dice', 0.0):.4f})")

    magfilo_dir = os.path.join(BASE_DATASET_PATH, "MAGFiLO_1.0_Kaggle_2026")
    test_img_dir = os.path.join(magfilo_dir, "test", "test_images")
    
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"Test image directory not found at {test_img_dir}")
        
    test_files = sorted(os.listdir(test_img_dir))
    print(f"Found {len(test_files)} test images in {test_img_dir}")
    
    submissions = []

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    print("\nGenerating Compressed COCO RLE Predictions for Test Images...")
    with torch.no_grad():
        for idx, fname in enumerate(test_files, 1):
            img_path = os.path.join(test_img_dir, fname)
            image_name = Path(fname).stem  # e.g. '20110120105534Ch'
            
            # Load and preprocess test image at 512x512
            pil_img = Image.open(img_path).convert('L')
            pil_img_resized = pil_img.resize(img_size, Image.BILINEAR)
            img_np_uint8 = np.array(pil_img_resized, dtype=np.uint8)
            img_np_clahe = clahe.apply(img_np_uint8)
            img_np = img_np_clahe.astype(np.float32) / 255.0
            
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
            
            # Forward pass
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Resize probability map back to full 2048 x 2048 resolution
            probs_full = cv2.resize(probs, orig_size, interpolation=cv2.INTER_LINEAR)
            binary_mask_full = (probs_full > prob_threshold).astype(np.uint8)

            # Find individual connected filament components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask_full)
            
            filament_count = 0
            for label in range(1, num_labels):  # Skip background (label 0)
                if stats[label, cv2.CC_STAT_AREA] < min_area_pixels:
                    continue
                    
                single_filament_mask = (labels == label).astype(np.uint8)
                coco_rle_str = mask_to_coco_rle(single_filament_mask)
                
                if coco_rle_str:
                    filament_count += 1
                    submissions.append({
                        'filament_id': f"{image_name}_{filament_count}",
                        'segmentation_rle': coco_rle_str
                    })
                    
            # Fallback: If no filaments detected, encode an empty background mask
            if filament_count == 0:
                empty_mask = np.zeros((orig_size[1], orig_size[0]), dtype=np.uint8)
                empty_rle_str = mask_to_coco_rle(empty_mask)
                submissions.append({
                    'filament_id': f"{image_name}_1",
                    'segmentation_rle': empty_rle_str
                })

            if idx % 30 == 0 or idx == len(test_files):
                print(f"  Processed {idx}/{len(test_files)} images (Total prediction rows: {len(submissions)})")

    # Construct Submission DataFrame
    sub_df = pd.DataFrame(submissions)

    print("\nVerification Check for NULL / NaN values:")
    print(sub_df.isnull().sum())
    print(f"\nTotal predictions generated: {len(sub_df)}")

    # 1. Save submission.csv directly in root and in submission_output/
    csv_root_path = "submission.csv"
    csv_output_path = os.path.join(output_dir, "submission.csv")
    
    sub_df.to_csv(csv_root_path, index=False)
    sub_df.to_csv(csv_output_path, index=False)
    print(f"[CSV] Generated: {csv_root_path} and {csv_output_path}")
    print("\nSample Preview (First 10 Rows):")
    print(sub_df.head(10))

    # 2. Save Zipped Submission Package
    zip_output_path = os.path.join(output_dir, "submission.zip")
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_output_path, os.path.basename(csv_output_path))
        
    print(f"\n[ZIP] Final Upload Archive created: {zip_output_path} ({os.path.getsize(zip_output_path)/1e6:.2f} MB)")

    print("=" * 75)
    print("SUCCESS! Kaggle submission file verified & generated with exact required schema:")
    print("Columns: ['filament_id', 'segmentation_rle']")
    print("Ready for direct Kaggle upload!")
    print("=" * 75)

if __name__ == "__main__":
    generate_submission()
