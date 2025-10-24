import torch
from torch.utils.data import Dataset
import glob
import cv2  # Use OpenCV to load PNG images
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# --- Albumentations Augmentation Pipeline ---
# This pipeline is strong to help reach the Dice target
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.Resize(height=256, width=256), # Resize all images to a consistent 256x256
    A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0), # Normalize images
    ToTensorV2(), # Convert to PyTorch Tensor
])

# A simpler pipeline for validation (no augmentation, just resize/normalize)
val_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
    ToTensorV2(),
])

class OASISDataset(Dataset):
    """
    PyTorch Dataset class for loading the 2D OASIS brain dataset.
    
    This version loads the PNG slices from the 'keras_png_slices_*' directories.
    """
    def __init__(self, data_path, mode='train', transform=None):
        """
        Args:
            data_path (str): The path to the main OASIS directory, 
                             e.g., /home/groups/comp3710/OASIS
            mode (str): 'train', 'validate', or 'test' to load the correct dataset.
            transform (callable, optional): Albumentations transform to be applied.
        """
        self.transform = transform
        
        # Set paths based on the mode
        if mode == 'train':
            self.image_dir = os.path.join(data_path, 'keras_png_slices_train')
            self.label_dir = os.path.join(data_path, 'keras_png_slices_seg_train')
        elif mode == 'validate':
            self.image_dir = os.path.join(data_path, 'keras_png_slices_validate')
            self.label_dir = os.path.join(data_path, 'keras_png_slices_seg_validate')
        elif mode == 'test':
            self.image_dir = os.path.join(data_path, 'keras_png_slices_test')
            self.label_dir = os.path.join(data_path, 'keras_png_slices_seg_test')
        
        # Get all filenames
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        self.label_files = sorted(glob.glob(os.path.join(self.label_dir, '*.png')))

        if not self.image_files or not self.label_files:
            print(f"Warning: No files found in {self.image_dir} or {self.label_dir}.")
        
        # Ensure the number of images and labels match
        assert len(self.image_files) == len(self.label_files), \
            f"Mismatch in file count: {len(self.image_files)} images, {len(self.label_files)} labels"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            # --- Load Image ---
            img_path = self.image_files[idx]
            # Load image using OpenCV. cv2.IMREAD_GRAYSCALE loads it as a 2D (H, W) array.
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError(f"Failed to load image: {img_path}")
            image = image.astype(np.float32)

            # --- Load Label ---
            label_path = self.label_files[idx]
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if label is None:
                raise IOError(f"Failed to load label: {label_path}")
            label = label.astype(np.float32)
            
            # Binarize label (making sure 0 is background, >0 is brain)
            label = (label > 0).astype(np.float32)

            # Apply augmentations
            if self.transform:
                augmented = self.transform(image=image, mask=label)
                image = augmented['image']
                label = augmented['mask']
            
            # Add a channel dimension for the mask (num_classes, H, W)
            label = label.unsqueeze(0)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading file {self.image_files[idx]} or {self.label_files[idx]}: {e}")
            # Return empty tensors on failure
            return torch.zeros((1, 256, 256)), torch.zeros((1, 256, 256))

if __name__ == "__main__":
    # This block allows you to test the dataset loader directly
    # python dataset.py
    
    DATA_PATH = "/home/groups/comp3710/OASIS"
    
    print("Testing 'train' dataset...")
    train_dataset = OASISDataset(DATA_PATH, mode='train', transform=transform)
    
    if len(train_dataset) > 0:
        print(f"Successfully loaded {len(train_dataset)} training files.")
        img, label = train_dataset[0]
        print(f"Image shape: {img.shape}")  # Should be [1, 256, 256]
        print(f"Label shape: {label.shape}") # Should be [1, 256, 256]
        print(f"Label unique values: {torch.unique(label)}")
    else:
        print("Train dataset test failed: No files were loaded.")

    print("\nTesting 'validate' dataset...")
    val_dataset = OASISDataset(DATA_PATH, mode='validate', transform=val_transform)
    if len(val_dataset) > 0:
        print(f"Successfully loaded {len(val_dataset)} validation files.")
    else:
        print("Validation dataset test failed.")

