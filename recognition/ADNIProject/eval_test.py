# eval_test.py
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import numpy as np

from dataset import get_adni_dataset
from modules import create_convnext_model

MODEL_SAVE_PATH = "best_adni_convnext.pth"
LOGIT_BIAS_PATH = "logit_bias.pt"
BATCH_SIZE = 64
NUM_CLASSES = 2

@torch.inference_mode()
def tta_logits(model, images):
    out = model(images)
    out += model(torch.flip(images, dims=[3]))
    for angle in (+7, -7):
        imgs = []
        for x in images:
            imgs.append(TF.rotate(x, angle=angle, interpolation=TF.InterpolationMode.BILINEAR, expand=False, fill=0))
        imgs = torch.stack(imgs, dim=0)
        out += model(imgs)
    return out / 4.0

@torch.inference_mode()
def evaluate(model, loader, device, logit_bias=0.0):
    model.eval()
    total = 0
    correct = 0
    cm = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = tta_logits(model, images)
        if logit_bias != 0.0:
            logits[:, 0] += logit_bias  # AD bias
        preds = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1
    acc = 100.0 * correct / total
    per_class = cm.diag() / cm.sum(dim=1).clamp(min=1)
    bacc = per_class.mean().item() * 100.0
    return acc, bacc, cm.cpu().numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_ds = get_adni_dataset('test')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = create_convnext_model(num_classes=NUM_CLASSES).to(device)
    state = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded weights: {MODEL_SAVE_PATH}")

    try:
        bias = float(torch.load(LOGIT_BIAS_PATH).item())
    except Exception:
        bias = 0.0
    print(f"Using AD logit bias: {bias:.3f}")

    acc, bacc, cm = evaluate(model, test_loader, device, logit_bias=bias)

    print("\n--- Final Test Results ---")
    print(f"Accuracy on Test Set: {acc:.2f}%")
    print(f"Balanced Accuracy on Test Set: {bacc:.2f}%")
    print("Confusion matrix (rows=true [AD, NC], cols=pred):")
    print(cm)

    if acc >= 80.0:
        print("\nProject target MET (>= 80%).")
    else:
        print("\nProject target NOT MET (>= 80%).")

if __name__ == "__main__":
    main()
