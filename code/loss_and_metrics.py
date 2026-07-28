import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Soft Dice Loss operating directly on logits with laplace smoothing.
    """
    def __init__(self, smooth=1e-5):
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
    Custom Loss combining Pos-Weighted BCEWithLogitsLoss and Soft Dice Loss.
    `pos_weight=5.0` heavily penalizes missing sparse solar filament pixels.
    """
    def __init__(self, bce_weight=0.4, dice_weight=0.6, pos_weight_val=5.0):
        super(CustomCombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        pos_weight_tensor = torch.tensor([pos_weight_val])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.dice = DiceLoss()

    def to(self, device):
        super().to(device)
        self.bce.pos_weight = self.bce.pos_weight.to(device)
        return self

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
        return total_loss, bce_loss, dice_loss

def calculate_metrics(logits, targets, threshold=0.4, smooth=1e-5):
    """
    Calculates Dice Score, IoU (Jaccard), Precision, and Recall at threshold 0.4.
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
