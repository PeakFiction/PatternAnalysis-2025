import torch
from torch.utils.data import Dataset
import glob
import nibabel as nib
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Albumentations Augmentation Pipeline ---
# Data augmentation is CRITICAL for reaching a 0.9 Dice score.
# This pipeline applies strong transformations.
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
    
    Assumes the data is in /home/groups/comp3710/OASIS/
    with subdirectories 'images' and 'labels'
    e.g., /home/groups/comp3710/OASIS/images/oasis_..._t1w.nii.gz
          /home/groups/comp3710/OASIS/labels/oasis_..._seg.nii.gz
    """
    def __init__(self, data_path, file_list, transform=None):
        self.transform = transform
        # Assumes image and label files share a common identifier
        # and are stored in respective folders.
        self.image_files = sorted(glob.glob(f"{data_path}/images/*.nii.gz"))
        self.label_files = sorted(glob.glob(f"{data_path}/labels/*.nii.gz"))

        # Use the provided file_list (indices) to select files for this dataset (train or val)
        self.image_files = [self.image_files[i] for i in file_list]
        self.label_files = [self.label_files[i] for i in file_list]

        if not self.image_files or not self.label_files:
            print(f"Warning: No files found in {data_path}/images or {data_path}/labels.")
            print(f"Searched for: {data_path}/images/*.nii.gz")
            
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            # --- Load Image ---
            img_path = self.image_files[idx]
            # Load Nifti file using nibabel
            image_nib = nib.load(img_path)
            # Get the image data as a numpy array
            image = image_nib.get_fdata().astype(np.float32)
            
            # --- Load Label ---
            label_path = self.label_files[idx]
            label_nib = nib.load(label_path)
            label = label_nib.get_fdata().astype(np.float32)
            
            # Ensure data is 2D (as per Appendix B logic)
            if image.ndim == 3 and image.shape[2] == 1:
                image = image[..., 0]
            if label.ndim == 3 and label.shape[2] == 1:
                label = label[..., 0]
                
            # Binarize label (assuming 0 is background, >0 is brain)
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
    # Test with all files (indices 0 to 9)
    test_indices = list(range(10)) 
    
    dataset = OASISDataset(DATA_PATH, file_list=test_indices, transform=transform)
    
    if len(dataset) > 0:
        print(f"Successfully loaded {len(dataset)} files.")
        img, label = dataset[0]
        print(f"Image shape: {img.shape}")   # Should be [1, 256, 256]
        print(f"Label shape: {label.shape}") # Should be [1, 256, 256]
        print(f"Image dtype: {img.dtype}")
        print(f"Label dtype: {label.dtype}")
        print(f"Image min/max: {img.min()}, {img.max()}")
        print(f"Label unique values: {torch.unique(label)}")
    else:
        print("Dataset test failed: No files were loaded.")
