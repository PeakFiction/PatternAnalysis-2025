import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from modules import UNet
from utils import dice_score

# --- Configuration ---
NUM_CLASSES = 6
PROSTATE_LABEL_INDEX = 5
BATCH_SIZE = 8
MODEL_SAVE_PATH = "best_hipmri_unet.pth"

def test():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Test Data ---
    print("Loading test dataset...")
    test_dataset = HipMRIDataset(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Found {len(test_dataset)} test images.")

    # --- 2. Load Best Model ---
    print(f"Loading best model from {MODEL_SAVE_PATH}...")
    model = UNet(n_channels=1, n_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    # --- 3. Run Evaluation ---
    print("Running final test on test set...")
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
        print("\nCongratulations! Project requirement MET. ")
    else:
        print("\nProject requirement NOT MET. ")

if __name__ == "__main__":
    test()
