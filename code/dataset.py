import os
import json
import numpy as np
# pyrefly: ignore [missing-import]
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw

class SolarFilamentDataset(Dataset):
    """
    PyTorch Dataset for Solar Filament Segmentation.
    Loads solar H-alpha / magnetogram images and converts COCO polygon annotations into binary segmentation masks.
    """
    def __init__(self, json_path, img_dir, img_size=(256, 256), is_train=True, val_split=0.15, seed=42):
        self.img_dir = img_dir
        self.img_size = img_size
        self.is_train = is_train
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotations file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            self.coco_raw = json.load(f)
            
        # Get disk images
        disk_files = set(os.listdir(img_dir)) if os.path.exists(img_dir) else set()
        
        # Build image records matching disk files
        valid_images = []
        for img in self.coco_raw['images']:
            fname = img['file_name']
            if fname in disk_files:
                valid_images.append(img)
            elif fname.replace('.jpeg', '.jpg') in disk_files:
                img['file_name'] = fname.replace('.jpeg', '.jpg')
                valid_images.append(img)
                
        # Deterministic Train / Validation Split
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(valid_images))
        val_count = int(len(valid_images) * val_split)
        
        if is_train:
            selected_indices = shuffled_indices[val_count:]
        else:
            selected_indices = shuffled_indices[:val_count]
            
        self.images = [valid_images[i] for i in selected_indices]
        
        # Build annotation mapping image_id -> list of annotations
        self.img_to_anns = {}
        for ann in self.coco_raw['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_record = self.images[idx]
        img_id = img_record['id']
        fname = img_record['file_name']
        img_path = os.path.join(self.img_dir, fname)
        
        # Load grayscale image
        pil_img = Image.open(img_path).convert('L')
        orig_w, orig_h = pil_img.size
        
        # Build binary mask at original resolution
        mask_pil = Image.new('L', (orig_w, orig_h), 0)
        draw = ImageDraw.Draw(mask_pil)
        
        anns = self.img_to_anns.get(img_id, [])
        for ann in anns:
            seg = ann.get('segmentation', [])
            for poly in seg:
                if len(poly) >= 6: # At least 3 2D coordinate pairs
                    draw.polygon(poly, fill=1)
                    
        # Resize image and mask to target training resolution
        pil_img_resized = pil_img.resize(self.img_size, Image.BILINEAR)
        mask_resized = mask_pil.resize(self.img_size, Image.NEAREST)
        
        # Convert to numpy arrays & normalize
        img_np = np.array(pil_img_resized, dtype=np.float32) / 255.0  # Range [0, 1]
        mask_np = np.array(mask_resized, dtype=np.float32)            # Range [0, 1]
        
        # Add channel dimension: shape [1, H, W]
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        
        return img_tensor, mask_tensor, fname

def get_dataloaders(json_path, img_dir, batch_size=8, img_size=(256, 256), val_split=0.15, num_workers=2):
    train_dataset = SolarFilamentDataset(
        json_path=json_path,
        img_dir=img_dir,
        img_size=img_size,
        is_train=True,
        val_split=val_split
    )
    val_dataset = SolarFilamentDataset(
        json_path=json_path,
        img_dir=img_dir,
        img_size=img_size,
        is_train=False,
        val_split=val_split
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))
    from analysis import path as BASE_DATASET_PATH
    
    magfilo_dir = os.path.join(BASE_DATASET_PATH, "MAGFiLO_1.0_Kaggle_2026")
    train_dir = os.path.join(magfilo_dir, "train")
    json_path = os.path.join(train_dir, "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    img_dir = os.path.join(train_dir, "train_images")
    
    train_loader, val_loader = get_dataloaders(json_path, img_dir, batch_size=4, img_size=(256, 256))
    print(f"Train Dataset size: {len(train_loader.dataset)}, Batches: {len(train_loader)}")
    print(f"Val Dataset size: {len(val_loader.dataset)}, Batches: {len(val_loader)}")
    
    for images, masks, fnames in train_loader:
        print(f"Sample Batch -> Images: {images.shape}, Masks: {masks.shape}, Range Img: [{images.min():.2f}, {images.max():.2f}]")
        break
