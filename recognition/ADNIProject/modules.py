import torch
import torch.nn as nn
import timm # PyTorch Image Models library

def create_convnext_model(num_classes=2, pretrained=True):
    """
    Loads a pretrained ConvNeXt model and replaces the classifier head.
    
    Args:
        num_classes (int): Number of output classes (AD vs NC = 2).
        pretrained (bool): Whether to load pretrained ImageNet weights.
    
    Returns:
        torch.nn.Module: The modified ConvNeXt model.
    """
    # Load the 'convnext_tiny' model pretrained on ImageNet
    # You can explore other variants like 'convnext_small', 'convnext_base', etc.
    # See timm documentation for available models: timm.list_models('convnext*')
    model = timm.create_model('convnext_tiny', pretrained=pretrained)
    
    # Get the number of input features for the classifier
    num_ftrs = model.head.fc.in_features
    
    # Replace the final fully connected layer (classifier head)
    # with a new one matching the number of classes in our dataset.
    model.head.fc = nn.Linear(num_ftrs, num_classes)
    
    print(f"Loaded ConvNeXt Tiny. Replaced classifier head for {num_classes} classes.")
    return model

# --- Quick test ---
if __name__ == '__main__':
    print("Testing model creation...")
    try:
        model = create_convnext_model(num_classes=2)
        print("Model created successfully.")
        # print(model) # Uncomment to see the full model structure
        
        # Test with dummy input
        dummy_input = torch.randn(4, 3, 224, 224) # Batch size 4, 3 channels, 224x224
        output = model(dummy_input)
        print(f"Output shape with dummy input: {output.shape}") # Should be [4, 2]
        
    except Exception as e:
        print(f"Model creation FAILED: {e}")
        print("Ensure 'timm' is installed: pip install timm")
