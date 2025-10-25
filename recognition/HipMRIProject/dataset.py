import os
import glob
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F  # Import interpolate function

# Define a fixed target size for all images and masks
TARGET_SIZE = (256, 128) # (Height, Width)

class HipMRIDataset(Dataset):
    """
    PyTorch Dataset for loading the 2D HipMRI slices.
    The data is expected to be in Nifti format, located on the Rangpur cluster.
    """
    def __init__(self, mode='train'):
        """
        Args:
            mode (str): One of 'train', 'validate', or 'test'
        """
        # Set the base data path on Rangpur
        self.data_root = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
        
        # Define image and mask directories based on the mode
        self.image_dir = os.path.join(self.data_root, f"keras_slices_{mode}")
        self.mask_dir = os.path.join(self.data_root, f"keras_slices_seg_{mode}")
        
        # Get sorted lists of all Nifti files (assuming .nii.gz format)
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.nii.gz")))
        self.mask_files = sorted(glob.glob(os.path.join(self.mask_dir, "*.nii.gz")))
        
        # Ensure the number of images and masks match
        assert len(self.image_files) == len(self.mask_files), \
            f"Found {len(self.image_files)} images but {len(self.mask_files)} masks in {mode}."
        assert len(self.image_files) > 0, f"No data found for mode '{mode}' in {self.data_root}"

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """Fetches a single image-mask pair."""
        
        # --- Load Image ---
        # Load the Nifti file using nibabel, as suggested in Appendix B
        img_nii = nib.load(self.image_files[idx])
        img = img_nii.get_fdata().astype(np.float32)
        
        # --- Load Mask ---
        mask_nii = nib.load(self.mask_files[idx])
        mask = mask_nii.get_fdata().astype(np.int64) # Use int64 for CrossEntropyLoss
        
        # --- Pre-processing ---
        # Handle extra dimensions, as hinted in Appendix B
        # (e.g., if data is saved as [H, W, 1], squeeze it to [H, W])
        if img.ndim == 3 and img.shape[2] == 1:
            img = np.squeeze(img, axis=2)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = np.squeeze(mask, axis=2)
            
        # Normalize image to [0, 1] range
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        
        # --- Convert to Tensors ---
        # Add a channel dimension for the image: [H, W] -> [1, H, W]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        
        # Mask should be [H, W] with Long type for the loss function
        mask_tensor = torch.from_numpy(mask).long()
        
        # --- !! NEW: Resize tensors to fixed size !! ---
        # F.interpolate needs a "batch" dimension, so we add and remove it.
        # Resize image: [1, H, W] -> [1, 1, H, W]
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), 
                                   size=TARGET_SIZE, 
                                   mode='bilinear', 
                                   align_corners=False)
        # Squeeze batch dim: [1, 1, 256, 128] -> [1, 256, 128]
        img_tensor = img_tensor.squeeze(0)
        
        # Resize mask: [H, W] -> [1, 1, H, W]
        # Use 'nearest' mode to avoid creating float values for labels (e.g., 2.5)
        mask_tensor = F.interpolate(mask_tensor.float().unsqueeze(0).unsqueeze(0), 
                                    size=TARGET_SIZE, 
                                    mode='nearest')
        # Squeeze dims: [1, 1, 256, 128] -> [256, 128] and convert back to long
        mask_tensor = mask_tensor.squeeze(0).squeeze(0).long()
        
        return img_tensor, mask_tensor

# --- Quick test to ensure it works ---
if __name__ == '__main__':
    print("Running a quick dataset test...")
    
    # Check training data
    try:
        train_dataset = HipMRIDataset(mode='train')
        print(f"Found {len(train_dataset)} training samples.")
        img, mask = train_dataset[0]
        print(f"Train Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Train Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Mask unique values: {torch.unique(mask)}")

        # Check validation data
        val_dataset = HipMRIDataset(mode='validate')
        print(f"\nFound {len(val_dataset)} validation samples.")
        img, mask = val_dataset[0]
        print(f"Validation Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Validation Mask shape: {mask.shape}, dtype: {mask.dtype}")
        
        # Check test data
        test_dataset = HipMRIDataset(mode='test')
        print(f"\nFound {len(test_dataset)} test samples.")
        img, mask = test_dataset[0]
        print(f"Test Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Test Mask shape: {mask.shape}, dtype: {mask.dtype}")

        print("\nDataset test passed!")
        
    except Exception as e:
        print(f"\nDataset test FAILED: {e}")
        print("Please check file paths and ensure 'nibabel' is installed in your conda env.")
        print("Install with: pip install nibabel")

