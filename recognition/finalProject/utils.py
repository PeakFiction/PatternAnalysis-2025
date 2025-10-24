import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1 - dice_score

class TverskyLoss(nn.Module):
    """
    Tversky Loss - a generalization of Dice Loss.
    Good for imbalanced data.
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        true_pos = (probs_flat * targets_flat).sum()
        false_neg = ((1 - probs_flat) * targets_flat).sum()
        false_pos = (probs_flat * (1 - targets_flat)).sum()
        
        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        
        return 1 - tversky_index


def dice_coefficient(logits, targets, smooth=1e-6):
    """
    Calculates the Dice Similarity Coefficient (DSC) metric.
    """
    # Apply sigmoid and threshold
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    
    # Flatten
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    dice = (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
    
    return dice.item() # Return as a Python number
