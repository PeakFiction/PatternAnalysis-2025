import torch
import torch.nn.functional as F

def dice_score(preds, targets, num_classes, smooth=1e-6):
    """
    Calculates the Dice score for each class in a multi-class segmentation.
    
    Args:
        preds (torch.Tensor): Model outputs (logits), shape [N, C, H, W]
        targets (torch.Tensor): Ground truth, shape [N, H, W]
        num_classes (int): Number of classes
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        torch.Tensor: A tensor of shape [C] containing the Dice score for each class.
    """
    # Get probabilities from logits
    preds_probs = F.softmax(preds, dim=1)
    
    # Get one-hot encoded predictions
    preds_one_hot = F.one_hot(preds_probs.argmax(dim=1), num_classes).permute(0, 3, 1, 2)
    
    # Get one-hot encoded targets
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2)
    
    # Calculate intersection and union for each class
    # Dims are [N, C, H, W]
    intersection = (preds_one_hot * targets_one_hot).sum(dim=[0, 2, 3])
    union = preds_one_hot.sum(dim=[0, 2, 3]) + targets_one_hot.sum(dim=[0, 2, 3])
    
    # Calculate Dice score per class
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice
