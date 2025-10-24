import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import glob
import os
import numpy as np

# Import from our other project files
from dataset import OASISDataset, transform, val_transform
from modules import ImprovedUNet
from utils import DiceLoss, dice_coefficient, TverskyLoss

# --- Configuration ---
DATA_PATH = "/home/groups/comp3710/OASIS"
BATCH_SIZE = 8
NUM_WORKERS = 4  # Number of CPU cores to use for data loading
EPOCHS = 100       # 0.9 Dice is hard, will need many epochs
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2    # 20% of data for validation
RANDOM_SEED = 42
MODEL_SAVE_PATH = "best_oasis_unet.pth"

def main():
    print(f"Starting OASIS UNet training...")
    print(f"Data path: {DATA_PATH}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Data Loading ---
    # Find all image files to create a train/val split
    try:
        all_files = sorted(glob.glob(f"{DATA_PATH}/images/*.nii.gz"))
        if not all_files:
            print(f"CRITICAL ERROR: No image files found in {DATA_PATH}/images/")
            print("Please check the path and directory structure.")
            return
        
        indices = np.arange(len(all_files))
        
        train_indices, val_indices = train_test_split(
            indices, test_size=VAL_SPLIT, random_state=RANDOM_SEED
        )
        
        print(f"Total files: {len(all_files)}")
        print(f"Training files: {len(train_indices)}")
        print(f"Validation files: {len(val_indices)}")
        
        train_dataset = OASISDataset(DATA_PATH, file_list=train_indices, transform=transform)
        val_dataset = OASISDataset(DATA_PATH, file_list=val_indices, transform=val_transform)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS, 
            pin_memory=True
        )
        
    except Exception as e:
        print(f"Error during data loading: {e}")
        return

    # --- Model, Loss, Optimizer ---
    model = ImprovedUNet(n_channels=1, n_classes=1).to(device)
    
    # A combined loss is best for segmentation
    # BCE is good for pixel-wise stability
    loss_bce = nn.BCEWithLogitsLoss()
    # Dice loss is good for handling class imbalance (small brain vs large background)
    loss_dice = DiceLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler() # For mixed-precision training (faster, less memory)
    
    best_val_dice = 0.0 # Track the best score

    # --- Training Loop ---
    print("Starting training loop...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # --- Forward pass with mixed precision ---
            with torch.cuda.amp.autocast():
                logits = model(images)
                # Combined Loss
                loss = loss_bce(logits, masks) + loss_dice(logits, masks)
            
            # --- Backward pass ---
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            if (i + 1) % 50 == 0: # Print update every 50 batches
                print(f"  Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # --- Validation Loop ---
        val_dice, avg_val_loss = evaluate(model, val_loader, loss_bce, loss_dice, device)
        
        print(f"Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Avg Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Dice Score: {val_dice:.4f}")
        
        # --- Save Best Model ---
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  *** New best model saved! Dice: {best_val_dice:.4f} ***")

    print("Training finished.")
    print(f"Best validation Dice score: {best_val_dice:.4f}")

def evaluate(model, loader, loss_bce, loss_dice, device):
    """Evaluation function to calculate validation loss and Dice score"""
    model.eval()
    total_dice = 0.0
    total_loss = 0.0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(images)
                # Calculate combined loss
                loss = loss_bce(logits, masks) + loss_dice(logits, masks)
                total_loss += loss.item()
            
            # Calculate Dice metric
            total_dice += dice_coefficient(logits, masks)
            
    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    return avg_dice, avg_loss

if __name__ == "__main__":
    main()
