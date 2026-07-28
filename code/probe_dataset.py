import os
import json
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image, ImageDraw

# Import dataset path from analysis.py
import sys
sys.path.append(os.path.dirname(__file__))
from analysis import path as BASE_DATASET_PATH

def probe():
    print("=" * 60)
    print("SOLAR FILAMENT DATASET PROBING")
    print("=" * 60)
    print(f"Base Dataset Path: {BASE_DATASET_PATH}")
    
    magfilo_dir = os.path.join(BASE_DATASET_PATH, "MAGFiLO_1.0_Kaggle_2026")
    train_dir = os.path.join(magfilo_dir, "train")
    test_dir = os.path.join(magfilo_dir, "test")
    
    json_path = os.path.join(train_dir, "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    train_img_dir = os.path.join(train_dir, "train_images")
    test_img_dir = os.path.join(test_dir, "test_images")
    
    if not os.path.exists(json_path):
        print(f"ERROR: Annotations JSON not found at {json_path}")
        return
        
    with open(json_path, 'r') as f:
        coco_raw = json.load(f)
        
    disk_train_files = set(os.listdir(train_img_dir)) if os.path.exists(train_img_dir) else set()
    print(f"\nTotal training images in COCO JSON: {len(coco_raw['images'])}")
    print(f"Total training image files on disk: {len(disk_train_files)}")
    
    # Filter valid images existing on disk
    valid_images = []
    for img in coco_raw['images']:
        fname = img['file_name']
        # check exact filename or with extension variations if needed
        if fname in disk_train_files:
            valid_images.append(img)
        elif fname.replace('.jpeg', '.jpg') in disk_train_files:
            img['file_name'] = fname.replace('.jpeg', '.jpg')
            valid_images.append(img)
            
    print(f"Verified images on disk matching JSON: {len(valid_images)}")
    
    # Build mapping from image_id to annotations
    img_to_anns = {}
    for ann in coco_raw['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
        
    # Inspect first matched image
    sample_img = valid_images[0]
    sample_img_id = sample_img['id']
    sample_filename = sample_img['file_name']
    sample_path = os.path.join(train_img_dir, sample_filename)
    
    img = Image.open(sample_path)
    img_np = np.array(img)
    print("\n--- Sample Image Analysis ---")
    print(f"  ID: {sample_img_id}")
    print(f"  File name: {sample_filename}")
    print(f"  Disk shape: {img_np.shape}, Mode: {img.mode}")
    print(f"  Min pixel value: {img_np.min()}, Max: {img_np.max()}, Mean: {img_np.mean():.2f}")
    
    sample_anns = img_to_anns.get(sample_img_id, [])
    print(f"  Annotations count: {len(sample_anns)}")
    
    # Rasterize polygons with PIL ImageDraw
    h, w = sample_img['height'], sample_img['width']
    mask_img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    
    for ann in sample_anns:
        seg = ann.get('segmentation', [])
        for poly in seg:
            # Flatten or coordinate pairs
            if len(poly) >= 6: # At least 3 points
                draw.polygon(poly, fill=1)
                
    mask_np = np.array(mask_img, dtype=np.uint8)
    print(f"  Rasterized Binary Mask shape: {mask_np.shape}")
    print(f"  Filament Foreground Pixels: {mask_np.sum()} ({mask_np.sum() / mask_np.size * 100:.3f}% of solar disc)")
    
    print("\n" + "=" * 60)
    print("Dataset Probing Completed Successfully!")

if __name__ == "__main__":
    probe()
