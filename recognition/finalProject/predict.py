import torch
import cv2 # Use OpenCV
import numpy as np
import glob
import os
from modules import ImprovedUNet
from dataset import val_transform # Use the validation transform (no augmentation)
import albumentations as A

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

    # Find test images from the 'test' directory
    test_image_dir = os.path.join(DATA_PATH, 'keras_png_slices_test')
    test_image_files = sorted(glob.glob(os.path.join(test_image_dir, '*.png')))
    
    if not test_image_files:
        print(f"No test images found in {test_image_dir}")
        return

    print(f"Running predictions on {len(test_image_files)} test images...")

    with torch.no_grad():
        for img_path in test_image_files:
            try:
                # --- Load and Preprocess ---
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise IOError(f"Failed to load image: {img_path}")
                
                image_data = image.astype(np.float32)
                original_shape = image.shape # (H, W)
                
                # Apply the same transformations as validation (resize, normalize, to_tensor)
                augmented = val_transform(image=image_data)
                image_tensor = augmented['image'].to(DEVICE)
                
                # Add batch dimension (B, C, H, W) -> (1, 1, 256, 256)
                image_tensor = image_tensor.unsqueeze(0)

                # --- Run Inference ---
                logits = model(image_tensor)
                
                # --- Post-process ---
                probs = torch.sigmoid(logits)
                pred_mask = (probs > 0.5).float()
                
                # Move to CPU, remove batch/channel dims, convert to numpy
                pred_mask_np = pred_mask.squeeze().cpu().numpy() # (256, 256)
                
                # --- Save Prediction ---
                # Resize the mask back to the original image size
                # We use OpenCV's resize with nearest neighbor interpolation
                pred_mask_resized = cv2.resize(
                    pred_mask_np, 
                    (original_shape[1], original_shape[0]), # (W, H) for cv2
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Ensure it's a 0-255 grayscale image for saving as PNG
                pred_mask_png = (pred_mask_resized * 255).astype(np.uint8)

                # Create a new Nifti image
                base_name = os.path.basename(img_path)
                output_name = os.path.join(OUTPUT_DIR, f"pred_{base_name}")
                
                cv2.imwrite(output_name, pred_mask_png)
                print(f"  Saved prediction to {output_name}")
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print("Prediction complete.")

if __name__ == "__main__":
    predict()

