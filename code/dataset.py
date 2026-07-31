import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import cv2
import albumentations as A


class SolarFilamentDataset(Dataset):
    """
    PyTorch Dataset for Solar Filament Segmentation.
    Loads solar H-alpha images and converts COCO polygon annotations into binary segmentation masks.
    Supports random horizontal/vertical flips and rotations for data augmentation.
    """
    def __init__(self, json_path, img_dir, img_size=(512, 512), is_train=True, val_split=0.15, seed=42):
        self.img_dir = img_dir
        self.img_size = img_size
        self.is_train = is_train
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotations file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            self.coco_raw = json.load(f)
            
        disk_files = set(os.listdir(img_dir)) if os.path.exists(img_dir) else set()
        
        valid_images = []
        for img in self.coco_raw['images']:
            fname = img['file_name']
            if fname in disk_files:
                valid_images.append(img)
            elif fname.replace('.jpeg', '.jpg') in disk_files:
                img['file_name'] = fname.replace('.jpeg', '.jpg')
                valid_images.append(img)
                
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(valid_images))
        val_count = int(len(valid_images) * val_split)
        
        if is_train:
            selected_indices = shuffled_indices[val_count:]
        else:
            selected_indices = shuffled_indices[:val_count]
            
        self.images = [valid_images[i] for i in selected_indices]
        
        self.img_to_anns = {}
        for ann in self.coco_raw['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        if self.is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.GaussNoise(p=0.2),
            ])
        else:
            self.transform = None
            
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_record = self.images[idx]
        img_id = img_record['id']
        fname = img_record['file_name']
        img_path = os.path.join(self.img_dir, fname)
        
        pil_img = Image.open(img_path).convert('L')
        orig_w, orig_h = pil_img.size
        
        mask_pil = Image.new('L', (orig_w, orig_h), 0)
        draw = ImageDraw.Draw(mask_pil)
        
        anns = self.img_to_anns.get(img_id, [])
        for ann in anns:
            seg = ann.get('segmentation', [])
            for poly in seg:
                if len(poly) >= 6:
                    draw.polygon(poly, fill=1)
                    
        pil_img_resized = pil_img.resize(self.img_size, Image.BILINEAR)
        mask_resized = mask_pil.resize(self.img_size, Image.NEAREST)
        
        img_np_uint8 = np.array(pil_img_resized, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_np_clahe = clahe.apply(img_np_uint8)
        
        img_np = img_np_clahe.astype(np.float32) / 255.0
        mask_np = np.array(mask_resized, dtype=np.float32)
        
        if self.transform is not None:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_np = augmented['image']
            mask_np = augmented['mask']
            
        img_tensor = torch.from_numpy(img_np.copy()).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np.copy()).unsqueeze(0)
        
        return img_tensor, mask_tensor, fname

def get_dataloaders(json_path, img_dir, batch_size=4, img_size=(512, 512), val_split=0.15, num_workers=2):
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
