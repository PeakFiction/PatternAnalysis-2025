import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import os
import numpy as np

# Import our custom modules
from dataset import get_adni_dataset # Assumes aggressive augmentation is still in dataset.py
from modules import create_convnext_model

# --- Configuration ---
NUM_CLASSES = 2
NUM_EPOCHS = 10 # Keep within 20 min limit
BATCH_SIZE = 32
LEARNING_RATE = 3e-5 # Back to the slightly lower LR
VALIDATION_SPLIT = 0.15
WEIGHT_DECAY = 0.01 # Keep moderate weight decay
LABEL_SMOOTHING = 0.1 # Add Label Smoothing

# Paths
MODEL_SAVE_PATH = "best_adni_convnext.pth"
PLOT_SAVE_PATH = "learning_curves_adni.png"

# --- Setup ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    print("Loading datasets...")
    full_train_dataset = get_adni_dataset(mode='train')
    test_dataset = get_adni_dataset(mode='test')

    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(VALIDATION_SPLIT * num_train))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = Subset(full_train_dataset, train_indices)
    temp_val_dataset_with_transforms = get_adni_dataset(mode='val')
    val_dataset = Subset(temp_val_dataset_with_transforms, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Split training data: {len(train_dataset)} train, {len(val_dataset)} validation.")
    print(f"Test data: {len(test_dataset)}.")

    # --- 2. Initialize Model, Loss, Optimizer ---
    print("Initializing model...")
    model = create_convnext_model(num_classes=NUM_CLASSES).to(device)

    # Loss function with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    print(f"Using CrossEntropyLoss with Label Smoothing: {LABEL_SMOOTHING}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    # --- 3. Training Loop ---
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # Use the same criterion for validation loss calculation
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.1e}")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved to {MODEL_SAVE_PATH} (Val Acc: {best_val_acc:.2f}%)")

    print("Training finished.")

    # --- 4. Plotting Results ---
    if plt:
      print(f"Saving learning curves to {PLOT_SAVE_PATH}...")
      plt.figure(figsize=(12, 5))

      plt.subplot(1, 2, 1)
      plt.plot(history['train_loss'], label='Train Loss')
      plt.plot(history['val_loss'], label='Val Loss')
      plt.title("Loss Curves")
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.legend()
      plt.grid(True)

      plt.subplot(1, 2, 2)
      plt.plot(history['train_acc'], label='Train Acc')
      plt.plot(history['val_acc'], label='Val Acc')
      plt.title("Accuracy Curves")
      plt.xlabel("Epoch")
      plt.ylabel("Accuracy (%)")
      plt.axhline(y=80.0, color='r', linestyle='--', label='Target (80%)')
      plt.legend()
      plt.grid(True)

      plt.tight_layout()
      plt.savefig(PLOT_SAVE_PATH)
      print("Plot saved.")

    # --- 5. Final Test ---
    print("\nRunning final test on test set using best model...")
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Loaded best model weights from {MODEL_SAVE_PATH} for final testing.")
    except FileNotFoundError:
        print(f"ERROR: Best model file not found at {MODEL_SAVE_PATH}. Testing with weights from the last epoch.")
        print("Cannot perform final test without a saved model.")
        return

    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_acc = 100 * correct_test / total_test

    print("\n--- Final Test Results ---")
    print(f"Accuracy on Test Set: {test_acc:.2f}% ({correct_test}/{total_test})")

    if test_acc >= 80.0:
        print("\nCongratulations! Project target accuracy (>= 80%) MET.")
    else:
        print("\nProject target accuracy (>= 80%) NOT MET. Further tuning or more epochs may be needed.")


if __name__ == "__main__":
    try:
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
    except ImportError:
       print("Matplotlib not found, skipping plot generation.")
       plt = None

    main()

