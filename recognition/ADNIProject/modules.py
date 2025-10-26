# modules.py
import torch
import torch.nn as nn
import timm  # PyTorch Image Models

def create_convnext_model(num_classes=2, pretrained=True):
    """
    ConvNeXt-Tiny with extra stochastic regularization and head dropout.
    """
    model = timm.create_model('convnext_tiny', pretrained=pretrained, drop_path_rate=0.2)
    num_ftrs = model.head.fc.in_features
    # add dropout before classifier to reduce overconfidence on shift
    model.head.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_ftrs, num_classes)
    )
    print(f"Loaded ConvNeXt Tiny. Replaced classifier head for {num_classes} classes.")
    return model

# --- Quick test ---
if __name__ == '__main__':
    print("Testing model creation...")
    try:
        model = create_convnext_model(num_classes=2)
        dummy_input = torch.randn(4, 3, 224, 224)
        output = model(dummy_input)
        print(f"Output shape with dummy input: {output.shape}")
    except Exception as e:
        print(f"Model creation FAILED: {e}")
        print("Ensure 'timm' is installed: pip install timm")
