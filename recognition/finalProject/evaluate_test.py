import torch
from torch.utils.data import DataLoader
import numpy as np
import os

# Import necessary components from your project files
from dataset import OASISDataset, val_transform # Use validation transform (no augmentation)
from modules import ImprovedUNet
from utils import dice_coefficient # Import the metric function

# --- Configuration ---
DATA_PATH = "/home/groups/comp3710/OASIS"
MODEL_PATH = "best_oasis_unet.pth" # Path to your best saved model
BATCH_SIZE = 16 # Can often use a larger batch size for evaluation
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_test_set():
    print("Starting evaluation on the TEST set...")
    print(f"Data path: {DATA_PATH}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Using device: {DEVICE}")

    # --- Load Test Dataset ---
    try:
        # Create dataset using 'test' mode
        test_dataset = OASISDataset(DATA_PATH, mode='test', transform=val_transform)
        print(f"Test files found: {len(test_dataset)}")
        
        if len(test_dataset) == 0:
             print("CRITICAL ERROR: Test dataset is empty. Check paths.")
             return

        test_loader = DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, # No shuffling needed for evaluation
            num_workers=NUM_WORKERS, 
            pin_memory=True
        )
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # --- Load Model ---
    model = ImprovedUNet(n_channels=1, n_classes=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
        print("Ensure the model was trained and saved correctly.")
        return
        
    model.eval() # Set model to evaluation mode

    # --- Evaluation Loop ---
    total_test_dice = 0.0
    num_batches = 0

    with torch.no_grad(): # Disable gradient calculations
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            
            # Use autocast for potential speedup, although less critical for inference
            with torch.amp.autocast(device_type=DEVICE.type if DEVICE.type != 'mps' else 'cpu', enabled=DEVICE.type=='cuda'):
                 logits = model(images)
            
            # Calculate Dice metric for this batch
            # Note: dice_coefficient expects logits, applies sigmoid/threshold internally
            batch_dice = dice_coefficient(logits, masks)
            total_test_dice += batch_dice
            num_batches += 1
            
            if (i + 1) % 20 == 0: # Print progress update
                 print(f"  Evaluated Batch {i+1}/{len(test_loader)}")

    # --- Calculate Average Test Dice ---
    avg_test_dice = total_test_dice / num_batches if num_batches > 0 else 0.0
    
    print("\n--- Test Set Evaluation Complete ---")
    print(f"Average Dice Score on Test Set: {avg_test_dice:.4f}")
    print("------------------------------------")
    
    # You should add this value to your README.md
    
if __name__ == "__main__":
    evaluate_test_set()

