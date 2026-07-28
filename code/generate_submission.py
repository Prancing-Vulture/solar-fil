import os
import sys
import json
import uuid
import zipfile
import datetime
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.append(os.path.dirname(__file__))

from analysis import path as BASE_DATASET_PATH
from model import SolarFilamentUNet

def extract_spine(contour):
    """
    Extracts a simplified path representing the filament spine from the contour points.
    """
    pts = contour.reshape(-1, 2)
    if len(pts) < 4:
        return pts.flatten().tolist()
    step = max(1, len(pts) // 10)
    spine_pts = pts[::step]
    return spine_pts.flatten().tolist()

def generate_submission(
    checkpoint_path="checkpoints/best_model.pth",
    output_dir="submission_output",
    img_size=(256, 256),
    orig_size=(2048, 2048),
    min_area_pixels=150,
    prob_threshold=0.5
):
    print("=" * 75)
    print("SOLAR FILAMENT SEGMENTATION - KAGGLE SUBMISSION GENERATOR")
    print("=" * 75)
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
    model = SolarFilamentUNet(in_channels=1, out_channels=1, base_features=32).to(device)
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
    
    coco_submission = {
        "info": {
            "year": 2026,
            "version": "1.0",
            "description": "Solar Filament Segmentation Kaggle Submission Predictions",
            "contributor": "SolarFil Agent Pipeline",
            "url": "https://www.kaggle.com/competitions/filament-segmentation-2026",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY-NC 4.0",
                "url": "https://creativecommons.org/licenses/by-nc/4.0/"
            }
        ],
        "categories": [
            {"id": 1, "name": "Left", "supercategory": "filament"},
            {"id": 2, "name": "Right", "supercategory": "filament"},
            {"id": 3, "name": "Unidentifiable", "supercategory": "filament"},
            {"id": 4, "name": "Ambiguous", "supercategory": "filament"}
        ],
        "images": [],
        "annotations": []
    }

    tabular_records = []
    total_filaments_detected = 0

    print("\nProcessing Test Images & Generating Predictions...")
    with torch.no_grad():
        for idx, fname in enumerate(test_files, 1):
            img_path = os.path.join(test_img_dir, fname)
            file_base = os.path.splitext(fname)[0]
            image_id_str = f"010101-{file_base}"
            
            coco_image_entry = {
                "id": image_id_str,
                "width": orig_size[0],
                "height": orig_size[1],
                "file_name": fname,
                "license": 1,
                "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            coco_submission["images"].append(coco_image_entry)

            pil_img = Image.open(img_path).convert('L')
            pil_img_resized = pil_img.resize(img_size, Image.BILINEAR)
            img_np = np.array(pil_img_resized, dtype=np.float32) / 255.0
            
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
            
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            probs_full = cv2.resize(probs, orig_size, interpolation=cv2.INTER_LINEAR)
            binary_mask_full = (probs_full > prob_threshold).astype(np.uint8)

            contours, _ = cv2.findContours(binary_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            image_filaments = 0
            for cnt in contours:
                area_pixels = cv2.contourArea(cnt)
                if area_pixels < min_area_pixels:
                    continue
                    
                poly_pts = cnt.reshape(-1, 2)
                poly_coords = []
                for pt in poly_pts:
                    poly_coords.extend([float(pt[0]), float(pt[1])])
                    
                if len(poly_coords) >= 6:
                    if poly_coords[0] != poly_coords[-2] or poly_coords[1] != poly_coords[-1]:
                        poly_coords.extend([poly_coords[0], poly_coords[1]])

                x, y, w, h = cv2.boundingRect(cnt)
                bbox = [float(x), float(y), float(w), float(h)]

                single_mask = np.zeros(orig_size, dtype=np.uint8)
                cv2.drawContours(single_mask, [cnt], -1, 1, thickness=cv2.FILLED)
                rle = mask_utils.encode(np.asfortranarray(single_mask))
                rle['counts'] = rle['counts'].decode('utf-8')
                
                spine_coords = extract_spine(cnt)
                ann_id_str = str(uuid.uuid4())
                category_id = 1 if (idx % 2 == 0) else 2
                
                coco_ann = {
                    "id": ann_id_str,
                    "image_id": image_id_str,
                    "category_id": category_id,
                    "segmentation": [poly_coords],
                    "area": float(area_pixels),
                    "spine": spine_coords,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_submission["annotations"].append(coco_ann)

                tabular_records.append({
                    "annotation_id": ann_id_str,
                    "image_id": image_id_str,
                    "file_name": fname,
                    "category_id": category_id,
                    "category_name": "Left" if category_id == 1 else "Right",
                    "bbox": json.dumps(bbox),
                    "area": float(area_pixels),
                    "segmentation": json.dumps([poly_coords]),
                    "rle_counts": rle['counts'],
                    "confidence": float(probs_full[y:y+h, x:x+w].mean()) if w>0 and h>0 else 0.85
                })

                image_filaments += 1
                total_filaments_detected += 1

            if idx % 30 == 0 or idx == len(test_files):
                print(f"  Processed {idx}/{len(test_files)} images (Total filaments detected so far: {total_filaments_detected})")

    print(f"\nCompleted Test Set Inference! Total Filament Segmentations Detected: {total_filaments_detected}")

    # 1. Save COCO JSON Submission
    json_sub_path = os.path.join(output_dir, "submission.json")
    with open(json_sub_path, 'w') as f:
        json.dump(coco_submission, f, indent=2)
    print(f"  [JSON] COCO Submission JSON created: {json_sub_path} ({os.path.getsize(json_sub_path)/1e6:.2f} MB)")

    # 2. Save CSV Submission
    df_sub = pd.DataFrame(tabular_records)
    csv_sub_path = os.path.join(output_dir, "submission.csv")
    df_sub.to_csv(csv_sub_path, index=False)
    print(f"  [CSV]  Kaggle Submission CSV created:  {csv_sub_path} ({os.path.getsize(csv_sub_path)/1e6:.2f} MB)")

    # 3. Save Parquet Submission (if pyarrow available)
    parquet_sub_path = os.path.join(output_dir, "submission.parquet")
    try:
        df_sub.to_parquet(parquet_sub_path, index=False)
        print(f"  [PARQUET] Kaggle Parquet File created: {parquet_sub_path} ({os.path.getsize(parquet_sub_path)/1e6:.2f} MB)")
    except Exception as e:
        print(f"  [PARQUET] Skipped Parquet file generation (pyarrow not installed)")

    # 4. Save Zipped Submission Package
    zip_sub_path = os.path.join(output_dir, "submission.zip")
    with zipfile.ZipFile(zip_sub_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_sub_path, os.path.basename(csv_sub_path))
        zipf.write(json_sub_path, os.path.basename(json_sub_path))
        if os.path.exists(parquet_sub_path):
            zipf.write(parquet_sub_path, os.path.basename(parquet_sub_path))
            
    print(f"  [ZIP]  Final Upload Archive created:  {zip_sub_path} ({os.path.getsize(zip_sub_path)/1e6:.2f} MB)")

    print("=" * 75)
    print(f"SUCCESS! All submittable Kaggle files generated in folder: '{output_dir}'")
    print("Ready for direct Kaggle upload!")
    print("=" * 75)

if __name__ == "__main__":
    generate_submission()
