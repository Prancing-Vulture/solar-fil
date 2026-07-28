# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Soft Dice Loss operating directly on logits.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice

class CustomCombinedLoss(nn.Module):
    """
    Custom Loss combining BCEWithLogitsLoss (logits level) and Soft Dice Loss.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super(CustomCombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
        return total_loss, bce_loss, dice_loss

def calculate_metrics(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Calculates Dice Score, IoU (Jaccard), Precision, and Recall given raw logits and targets.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        tp = (preds_flat * targets_flat).sum()
        fp = (preds_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - preds_flat) * targets_flat).sum()
        
        dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
        iou = (tp + smooth) / (tp + fp + fn + smooth)
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

if __name__ == "__main__":
    logits = torch.randn(4, 1, 256, 256)
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()
    
    criterion = CustomCombinedLoss()
    total_loss, bce, dice = criterion(logits, targets)
    metrics = calculate_metrics(logits, targets)
    
    print(f"Custom Combined Loss: {total_loss.item():.4f}")
    print(f"  -> BCE Logits Loss: {bce.item():.4f}")
    print(f"  -> Dice Loss: {dice.item():.4f}")
    print(f"Calculated Metrics: {metrics}")
