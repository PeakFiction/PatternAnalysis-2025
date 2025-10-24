import torch
import nibabel as nib
import numpy as np
import glob
import os
from modules import ImprovedUNet
from dataset import val_transform # Use the validation transform (no augmentation)
from scipy.ndimage import zoom # Make sure scipy is installed

# --- Configuration ---
DATA_PATH = "/home/groups/comp3710/OASIS"
MODEL_PATH = "best_oasis_unet.pth" # Path to your saved model
OUTPUT_DIR = "predictions"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict():
    print(f"Loading model from {MODEL_PATH}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize model and load weights
    model = ImprovedUNet(n_channels=1, n_classes=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
        print("Please make sure you have successfully trained the model and the .pth file exists.")
        return
        
    model.eval() # Set model to evaluation mode

    # Find a few test images
    # Let's just grab the first 5 images from the main images folder
    test_image_files = sorted(glob.glob(f"{DATA_PATH}/images/*.nii.gz"))[:5]
    
    if not test_image_files:
        print(f"No test images found in {DATA_PATH}/images/")
        return

    print(f"Running predictions on {len(test_image_files)} images...")

    with torch.no_grad():
        for img_path in test_image_files:
            try:
                # --- Load and Preprocess ---
                img_nib = nib.load(img_path)
                img_data = img_nib.get_fdata().astype(np.float32)
                
                # Save original affine/header to save the prediction correctly
                original_affine = img_nib.affine
                original_header = img_nib.header

                # Ensure data is 2D
                if img_data.ndim == 3 and img_data.shape[2] == 1:
                    img_data = img_data[..., 0]
                
                original_shape = (img_data.shape[0], img_data.shape[1])
                
                # Apply the same transformations as validation (resize, normalize, to_tensor)
                # Note: 'mask' isn't needed here, so we just pass the image
                augmented = val_transform(image=img_data)
                image_tensor = augmented['image'].to(DEVICE)
                
                # Add batch dimension (B, C, H, W) -> (1, 1, 256, 256)
                image_tensor = image_tensor.unsqueeze(0)

                # --- Run Inference ---
                logits = model(image_tensor)
                
                # --- Post-process ---
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                # Threshold at 0.5 to get binary mask
                pred_mask = (probs > 0.5).float()
                
                # Move to CPU, remove batch/channel dims, convert to numpy
                pred_mask_np = pred_mask.squeeze().cpu().numpy()
                
                # --- Save Prediction ---
                # We need to resize the mask back to the original image size
                # (The val_transform resized it to 256x256)
                zoom_factors = (original_shape[0] / 256, original_shape[1] / 256)
                pred_mask_resized = zoom(pred_mask_np, zoom_factors, order=0) # order=0 is nearest neighbor
                
                # Ensure it's integer type for segmentation mask
                pred_mask_resized = pred_mask_resized.astype(np.uint8)

                # Create a new Nifti image
                base_name = os.path.basename(img_path)
                output_name = os.path.join(OUTPUT_DIR, f"pred_{base_name}")
                
                pred_nib = nib.Nifti1Image(pred_mask_resized, original_affine, original_header)
                nib.save(pred_nib, output_name)
                
                print(f"  Saved prediction to {output_name}")
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print("Prediction complete.")

if __name__ == "__main__":
    predict()

