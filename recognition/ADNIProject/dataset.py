import os
from torchvision import datasets, transforms

# --- Configuration ---
DATA_ROOT = "/home/groups/comp3710/ADNI/AD_NC"
IMAGE_SIZE = 224  # ConvNeXt input size

class PerImageZScore(object):
    def __call__(self, tensor):
        # tensor: C x H x W in [0,1]
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-6)

def get_adni_transforms(mode='train'):
    """
    ADNI MRI are effectively grayscale. Force grayscale->3ch,
    use per-image z-score; avoid ImageNet stats.
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),      # left/right invariance
            transforms.RandomRotation(10, fill=0),       # mild rotation only
            transforms.ToTensor(),
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
    'val' uses the same directory as 'train' but eval transforms;
    indices are split in train.py.
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
