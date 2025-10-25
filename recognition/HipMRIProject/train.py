import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Import our custom modules
from dataset import HipMRIDataset
from modules import UNet
from utils import dice_score

# --- Configuration ---
# Data and Model Hyperparameters
NUM_CLASSES = 6  # 0: BG, 1: Body, 2: Bone, 3: Bladder, 4: Rectum, 5: Prostate
PROSTATE_LABEL_INDEX = 5 # As per our assumption
NUM_EPOCHS = 25
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

# Paths
MODEL_SAVE_PATH = "best_hipmri_unet.pth"
PLOT_SAVE_PATH = "learning_curves.png"

# --- Setup ---
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    print("Loading datasets...")
    train_dataset = HipMRIDataset(mode='train')
    val_dataset = HipMRIDataset(mode='validate')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Found {len(train_dataset)} train images and {len(val_dataset)} validation images.")

    # --- 2. Initialize Model, Loss, Optimizer ---
    print("Initializing model...")
    # Model is 1 input channel (grayscale) and NUM_CLASSES output channels (logits)
    model = UNet(n_channels=1, n_classes=NUM_CLASSES).to(device)
    
    # Loss function (for multi-class segmentation)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training Loop ---
    best_val_dice = -1.0
    history = {'train_loss': [], 'val_loss': [], 'val_prostate_dice': []}

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_train_loss = 0.0
        
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device) # [N, 1, H, W], [N, H, W]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images) # [N, C, H, W]
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Validation Loop ---
        model.eval()
        running_val_loss = 0.0
        all_val_dice_scores = torch.zeros(NUM_CLASSES).to(device)
        num_val_batches = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item()
                
                # Calculate metrics
                all_val_dice_scores += dice_score(outputs, masks, NUM_CLASSES)
                num_val_batches += 1
        
        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_dice = all_val_dice_scores / num_val_batches
        
        val_prostate_dice = avg_val_dice[PROSTATE_LABEL_INDEX].item()
        history['val_loss'].append(avg_val_loss)
        history['val_prostate_dice'].append(val_prostate_dice)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Prostate Dice: {val_prostate_dice:.4f}")

        # Save the best model based on prostate dice score
        if val_prostate_dice > best_val_dice:
            best_val_dice = val_prostate_dice
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved to {MODEL_SAVE_PATH} (Dice: {best_val_dice:.4f})")

    print("Training finished.")

    # --- 4. Plotting Results ---
    print(f"Saving learning curves to {PLOT_SAVE_PATH}...")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_prostate_dice'], label='Prostate Dice', color='orange')
    plt.title("Validation Prostate Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.axhline(y=0.75, color='r', linestyle='--', label='Target (0.75)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    print("Plot saved.")

    # --- 5. Final Test ---
    print("Running final test on test set...")
    test_dataset = HipMRIDataset(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    all_test_dice_scores = torch.zeros(NUM_CLASSES).to(device)
    num_test_batches = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            all_test_dice_scores += dice_score(outputs, masks, NUM_CLASSES)
            num_test_batches += 1
            
    avg_test_dice = all_test_dice_scores / num_test_batches
    
    print("\n--- Final Test Results ---")
    print(f"Average Dice Scores per class:")
    labels = ["BG", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
    for i, label in enumerate(labels):
        print(f"  {label}: {avg_test_dice[i].item():.4f}")
        
    prostate_dice_final = avg_test_dice[PROSTATE_LABEL_INDEX].item()
    print(f"\n**Final Prostate Dice on Test Set: {prostate_dice_final:.4f}**")
    
    if prostate_dice_final >= 0.75:
        print("Congratulations! Project target MET.")
    else:
        print("Project target NOT MET. Keep tuning!")


if __name__ == "__main__":
    main()
