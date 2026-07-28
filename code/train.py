import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from analysis import path as BASE_DATASET_PATH
from dataset import get_dataloaders
from model import SolarFilamentUNet
from loss_and_metrics import CustomCombinedLoss, calculate_metrics
from displayer import LiveDisplayer

def train_model(num_epochs=10, batch_size=8, lr=1e-3, img_size=(256, 256), save_dir="checkpoints"):
    print("=" * 75)
    print("SOLAR FILAMENT SEGMENTATION - LIVE TRAINING & DISPLAYER PIPELINE")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")

    magfilo_dir = os.path.join(BASE_DATASET_PATH, "MAGFiLO_1.0_Kaggle_2026")
    train_dir = os.path.join(magfilo_dir, "train")
    json_path = os.path.join(train_dir, "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    img_dir = os.path.join(train_dir, "train_images")
    
    print(f"Dataset Location: {img_dir}")
    train_loader, val_loader = get_dataloaders(
        json_path=json_path,
        img_dir=img_dir,
        batch_size=batch_size,
        img_size=img_size,
        val_split=0.15
    )
    print(f"Train Dataset: {len(train_loader.dataset)} images ({len(train_loader)} batches)")
    print(f"Validation Dataset: {len(val_loader.dataset)} images ({len(val_loader)} batches)")
    
    model = SolarFilamentUNet(in_channels=1, out_channels=1, base_features=32).to(device)
    criterion = CustomCombinedLoss(bce_weight=0.5, dice_weight=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    # Launch Live Displayer Server
    displayer = LiveDisplayer(port=5000)
    print("Live Web Displayer active at: http://localhost:5000")
    print("Live Epoch Plot updated at:  live_display/live_epoch_display.png")
    
    # Pick fixed validation sample for live epoch segmentation visualization
    fixed_val_img, fixed_val_mask, fixed_val_fname = val_loader.dataset[0]
    fixed_val_img_device = fixed_val_img.unsqueeze(0).to(device)
    
    best_val_dice = 0.0

    print("\nStarting Training & Live Metric Logging...\n")
    for epoch in range(1, num_epochs + 1):
        # TRAIN
        model.train()
        train_loss_sum = 0.0
        train_dice_sum = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{num_epochs:02d} [Train]", leave=False)
        for images, masks, _ in train_pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            total_loss, bce_loss, dice_loss = criterion(logits, masks)
            
            total_loss.backward()
            optimizer.step()
            
            train_loss_sum += total_loss.item()
            m = calculate_metrics(logits, masks)
            train_dice_sum += m['dice']
            train_pbar.set_postfix({'loss': f"{total_loss.item():.4f}", 'dice': f"{m['dice']:.4f}"})
            
        scheduler.step()
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_dice = train_dice_sum / len(train_loader)

        # VALIDATION
        model.eval()
        val_loss_sum = 0.0
        val_bce_sum = 0.0
        val_dice_sum = 0.0
        val_iou_sum = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{num_epochs:02d} [Val]  ", leave=False)
            for images, masks, _ in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                logits = model(images)
                total_loss, bce_loss, dice_loss = criterion(logits, masks)
                
                val_loss_sum += total_loss.item()
                val_bce_sum += bce_loss.item()
                
                m = calculate_metrics(logits, masks)
                val_dice_sum += m['dice']
                val_iou_sum += m['iou']
                val_pbar.set_postfix({'val_loss': f"{total_loss.item():.4f}", 'val_dice': f"{m['dice']:.4f}"})
                
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_bce = val_bce_sum / len(val_loader)
        avg_val_dice = val_dice_sum / len(val_loader)
        avg_val_iou = val_iou_sum / len(val_loader)

        # LIVE PREDICTION FOR DISPLAYER
        with torch.no_grad():
            sample_logits = model(fixed_val_img_device)
            sample_probs = torch.sigmoid(sample_logits).squeeze().cpu().numpy()
            
        sample_img_np = fixed_val_img.squeeze().numpy()
        sample_gt_np = fixed_val_mask.squeeze().numpy()

        displayer.update_epoch(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            train_dice=avg_train_dice,
            val_dice=avg_val_dice,
            val_iou=avg_val_iou,
            val_bce=avg_val_bce,
            sample_img_np=sample_img_np,
            sample_gt_np=sample_gt_np,
            sample_pred_probs=sample_probs
        )

        # Print ASCII Epoch Summary Box
        print("+" + "-" * 73 + "+")
        print(f"| EPOCH {epoch:02d}/{num_epochs:02d} METRICS REPORT                                              |")
        print("+" + "-" * 73 + "+")
        print(f"|  Train Loss (BCE+Dice): {avg_train_loss:.4f}  |  Val Loss (BCE+Dice): {avg_val_loss:.4f}        |")
        print(f"|  Train Dice Score:       {avg_train_dice:.4f}  |  Val Dice Score:      {avg_val_dice:.4f}        |")
        print(f"|  Val BCE Logits Loss:    {avg_val_bce:.4f}  |  Val IoU (Jaccard):   {avg_val_iou:.4f}        |")
        print("+" + "-" * 73 + "+")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': avg_val_dice,
                'val_loss': avg_val_loss,
            }, best_model_path)
            print(f"|  [*] NEW BEST VAL DICE: {best_val_dice:.4f} (Saved to {best_model_path}) |")
        else:
            print(f"|  Best Val Dice So Far: {best_val_dice:.4f}                                         |")
        print("+" + "-" * 73 + "+\n")

    print("=" * 75)
    print(f"[COMPLETE] TRAINING COMPLETED! Best Validation Dice Score: {best_val_dice:.4f}")
    print(f"Checkpoint saved: {best_model_path}")
    print(f"Final Display Image: live_display/live_epoch_display.png")
    print("=" * 75)

if __name__ == "__main__":
    train_model(num_epochs=10, batch_size=8, lr=1e-3)
