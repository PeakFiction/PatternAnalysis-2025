# dataset.py
import os
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

# --- Configuration ---
DATA_ROOT = "/home/groups/comp3710/ADNI/AD_NC"
IMAGE_SIZE = 224  # ConvNeXt input size

# ---- MRI-friendly transforms ----
class PerImageZScore(object):
    def __call__(self, tensor):
        # tensor: C x H x W, assumed in [0,1] before z-scoring
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-6)

class AdditiveGaussianNoise(object):
    def __init__(self, std_min=0.005, std_max=0.03):
        self.std_min = std_min
        self.std_max = std_max
    def __call__(self, tensor):
        # tensor in [0,1]
        std = torch.empty(1).uniform_(self.std_min, self.std_max).item()
        noise = torch.randn_like(tensor) * std
        x = tensor + noise
        return x.clamp(0.0, 1.0)

class RandomGamma(object):
    def __init__(self, gamma_min=0.85, gamma_max=1.15, p=0.5):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.p = p
    def __call__(self, tensor):
        # tensor in [0,1]
        if torch.rand(1).item() < self.p:
            gamma = torch.empty(1).uniform_(self.gamma_min, self.gamma_max).item()
            return TF.adjust_gamma(tensor, gamma=gamma, gain=1.0)
        return tensor

def get_adni_transforms(mode='train'):
    """
    ADNI slices are effectively grayscale. Force grayscale->3ch,
    use per-image z-score normalization (no ImageNet stats).
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.90, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7, fill=0),
            transforms.ToTensor(),
            AdditiveGaussianNoise(0.005, 0.02),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),  # mild intensity shifts
            RandomGamma(0.9, 1.1, p=0.5),
            PerImageZScore(),
        ])
    else:  # 'val' or 'test'
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            PerImageZScore(),
        ])

def get_adni_dataset(mode='train'):
    """
    Loads the ADNI dataset using ImageFolder.

    Args:
        mode (str): 'train', 'val', or 'test'
    Returns:
        torchvision.datasets.ImageFolder
    """
    if mode == 'val':
        data_dir = os.path.join(DATA_ROOT, 'train')
        transform = get_adni_transforms('val')
    elif mode == 'test':
        data_dir = os.path.join(DATA_ROOT, 'test')
        transform = get_adni_transforms('test')
    else:  # 'train'
        data_dir = os.path.join(DATA_ROOT, 'train')
        transform = get_adni_transforms('train')

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    if len(dataset) == 0:
        raise ValueError(f"No images found in {data_dir}. Check dataset paths and structure.")

    print(f"Loaded {len(dataset)} images for mode '{mode}' from {data_dir}")
    print(f"Class mapping: {dataset.class_to_idx}")

    return dataset

# --- Quick test ---
if __name__ == '__main__':
    print("Running dataset loading test...")
    try:
        train_ds = get_adni_dataset('train')
        test_ds = get_adni_dataset('test')
        img, label = train_ds[0]
        print(f"\nSample Train Image shape: {img.shape}, dtype: {img.dtype}, Label: {label}")
        img, label = test_ds[0]
        print(f"Sample Test Image shape: {img.shape}, dtype: {img.dtype}, Label: {label}")
        print("\nDataset test PASSED!")
    except Exception as e:
        print(f"\nDataset test FAILED: {e}")
        print("Ensure the DATA_ROOT path is correct and torchvision/Pillow are installed.")
        print("Install with: pip install torchvision timm Pillow")
