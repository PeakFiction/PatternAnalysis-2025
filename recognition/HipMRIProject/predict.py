import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Import our custom modules
from dataset import HipMRIDataset
from modules import UNet

# --- Configuration ---
NUM_CLASSES = 6
MODEL_SAVE_PATH = "best_hipmri_unet.pth"
OUTPUT_IMAGE_NAME = "prediction_visualization.png"

# Select a sample from the test set to visualize
# You can change this number to see different predictions
SAMPLE_IDX_TO_VISUALIZE = 15

def predict_and_visualize():
    """
    Loads the trained model, runs prediction on a single test sample,
    and saves a 3-panel visualization.
    """
    
    # Use a local variable for the sample index, initialized 
    # from the global constant. This avoids the UnboundLocalError.
    sample_idx = SAMPLE_IDX_TO_VISUALIZE
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    print(f"Loading model from {MODEL_SAVE_PATH}...")
    model = UNet(n_channels=1, n_classes=NUM_CLASSES)
    
    # Load weights onto the correct device
    try:
        # Use map_location to ensure it works even if you run this on a CPU
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}")
        print("Please run train.py first to generate the model file.")
        return
    except Exception as e:
        print(f"Error loading model state: {e}")
        return
        
    model.to(device)
    model.eval() # Set model to evaluation mode (e.g., disables dropout)

    # --- 2. Load a Sample from Test Set ---
    print(f"Loading test sample index {sample_idx}...")
    try:
        test_dataset = HipMRIDataset(mode='test')
        if sample_idx >= len(test_dataset):
            print(f"Warning: Sample index {sample_idx} is out of range (max is {len(test_dataset)-1}).")
            print("Using index 0 instead.")
            sample_idx = 0
        
        # Get the single sample. 
        # Shapes are already [1, H, W] and [H, W] from our dataset.py
        image_tensor, mask_tensor = test_dataset[sample_idx]
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Please check the data path in dataset.py: {test_dataset.data_root}")
        return

    # --- 3. Prepare for Prediction ---
    # Add a batch dimension: [1, H, W] -> [1, 1, H, W]
    # Send to the same device as the model
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # --- 4. Run Prediction ---
    print("Running model prediction...")
    with torch.no_grad(): # Disable gradient calculation for inference
        output_logits = model(input_tensor) # Output shape: [1, C, H, W]
    
    # Get the predicted class (index) for each pixel
    # .argmax(dim=1) finds the class with the max logit value
    # Resulting shape is [1, H, W]
    # .squeeze(0) removes the batch dim -> [H, W]
    # .cpu().numpy() moves it to CPU and converts to NumPy array
    pred_mask = output_logits.argmax(dim=1).squeeze(0).cpu().numpy()

    # --- 5. Prepare Tensors for Plotting ---
    # Original image: [1, H, W] -> [H, W]
    original_image = image_tensor.squeeze(0).cpu().numpy()
    # Ground truth mask: [H, W] (already in correct shape)
    ground_truth_mask = mask_tensor.cpu().numpy()

    # --- 6. Create and Save Visualization ---
    print(f"Saving visualization to {OUTPUT_IMAGE_NAME}...")
    
    # Define class labels for the colorbar
    labels = ["BG", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
    
    # Use a colormap that has a distinct color for 0 (background)
    # 'nipy_spectral' is good (0=black)
    cmap = plt.cm.get_cmap('nipy_spectral', NUM_CLASSES)
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot Original Image
    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title(f"Original Image (Sample {sample_idx})")
    ax[0].axis('off')
    
    # Plot Ground Truth Mask
    im1 = ax[1].imshow(ground_truth_mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
    ax[1].set_title("Ground Truth Mask")
    ax[1].axis('off')
    
    # Plot Predicted Mask
    im2 = ax[2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
    ax[2].set_title("Predicted Mask (Dice: 0.8362)")
    ax[2].axis('off')
    
    # Add a shared colorbar
    fig.subplots_adjust(right=0.85) # Make room for colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_ticks(np.arange(NUM_CLASSES) + 0.5 * (NUM_CLASSES-1)/NUM_CLASSES)
    cbar.set_ticklabels(labels)
    
    plt.savefig(OUTPUT_IMAGE_NAME, bbox_inches='tight')
    
    print(f"Successfully saved {OUTPUT_IMAGE_NAME}")

if __name__ == "__main__":
    predict_and_visualize()


