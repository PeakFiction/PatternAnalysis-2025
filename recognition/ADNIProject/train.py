import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

from dataset import get_adni_dataset
from modules import create_convnext_model

# --- Configuration ---
NUM_CLASSES = 2
NUM_EPOCHS = 10            # keep runtime reasonable
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
VALIDATION_SPLIT = 0.15
WEIGHT_DECAY = 0.01
MODEL_SAVE_PATH = "best_adni_convnext.pth"
PLOT_SAVE_PATH = "learning_curves_adni.png"
SEED = 42

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def build_weighted_sampler(full_train_dataset, train_indices):
    """Handle possible class imbalance via per-sample weights."""
    # ImageFolder stores (path, class_idx) in .samples
    targets = np.array([full_train_dataset.samples[i][1] for i in train_indices], dtype=np.int64)
    class_counts = np.bincount(targets, minlength=NUM_CLASSES).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

@torch.inference_mode()
def eval_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        # light TTA: identity + horizontal flip
        logits = model(images)
        logits += model(torch.flip(images, dims=[3]))
        preds = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return 100.0 * correct / total

def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    print("Loading datasets...")
    full_train_dataset = get_adni_dataset(mode='train')
    test_dataset = get_adni_dataset(mode='test')

    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(VALIDATION_SPLIT * num_train))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = Subset(full_train_dataset, train_indices)
    temp_val_dataset_with_transforms = get_adni_dataset(mode='val')
    val_dataset = Subset(temp_val_dataset_with_transforms, val_indices)

    sampler = build_weighted_sampler(full_train_dataset, train_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"Split training data: {len(train_dataset)} train, {len(val_dataset)} validation.")
    print(f"Test data: {len(test_dataset)}.")

    # --- 2. Model / Loss / Optim ---
    print("Initializing model...")
    model = create_convnext_model(num_classes=NUM_CLASSES).to(device)

    # label smoothing fights overconfidence on distribution shift
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    # --- 3. Train ---
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        # Validation (no TTA; keep it comparable and fast)
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = 100.0 * correct_val / total_val
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.1e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved to {MODEL_SAVE_PATH} (Val Acc: {best_val_acc:.2f}%)")

    print("Training finished.")

    # --- 4. Plot ---
    if plt:
        print(f"Saving learning curves to {PLOT_SAVE_PATH}...")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # re-import for backend
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

    # --- 5. Final Test with light TTA ---
    print("\nRunning final test on test set using best model...")
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded best model weights from {MODEL_SAVE_PATH} for final testing.")
    except FileNotFoundError:
        print(f"ERROR: Best model file not found at {MODEL_SAVE_PATH}. Testing with weights from the last epoch.")
        return

    test_acc = eval_accuracy(model, test_loader, device)

    print("\n--- Final Test Results ---")
    print(f"Accuracy on Test Set: {test_acc:.2f}%")

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
