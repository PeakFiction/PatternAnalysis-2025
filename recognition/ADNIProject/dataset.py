import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Configuration ---
DATA_ROOT = "/home/groups/comp3710/ADNI/AD_NC"
IMAGE_SIZE = 224 # ConvNeXt standard input size
# Standard ImageNet normalization values
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def get_adni_transforms(mode='train'):
    """
    Returns the appropriate transforms.
    Includes aggressive augmentation for the training set.
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)), # Allow cropping more area
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30), # Increased rotation
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=15), # Added shear/translation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15), # Increased jitter
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])
    else: # 'val' or 'test'
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])

def get_adni_dataset(mode='train'):
    """
    Loads the ADNI dataset using ImageFolder.

    Args:
        mode (str): 'train', 'val', or 'test'
    Returns:
        torchvision.datasets.ImageFolder: The loaded dataset.
    """
    if mode == 'val':
        data_dir = os.path.join(DATA_ROOT, 'train')
        transform = get_adni_transforms('val')
    elif mode == 'test':
        data_dir = os.path.join(DATA_ROOT, 'test')
        transform = get_adni_transforms('test')
    else: # 'train'
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
