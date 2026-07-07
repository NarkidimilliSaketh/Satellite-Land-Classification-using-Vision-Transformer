import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from config import Config

def create_model(num_classes=Config.NUM_CLASSES, freeze_backbone=True):
    """
    Creates and returns the ViT-B/16 model.
    By default, it uses transfer learning by freezing the backbone and replacing the head.
    """
    print("Loading pretrained ViT-B/16 (IMAGENET1K_V1)...")
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    
    if freeze_backbone:
        print("Freezing ViT backbone...")
        for param in model.parameters():
            param.requires_grad = False
            
    # Replace the classification head
    # ViT-B/16 in torchvision has a `heads` attribute which is a Sequential containing a Linear layer
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    # The new head will automatically have requires_grad=True
    return model

def unfreeze_backbone(model):
    """
    Unfreezes the entire model backbone for fine-tuning.
    """
    print("Unfreezing ViT backbone for fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True
    return model

def generate_model_summary(model):
    import json
    import os
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    summary = {
        "Model Name": Config.MODEL_NAME,
        "Backbone": "ViT-B/16 (IMAGENET1K_V1)",
        "Number of Parameters": total_params,
        "Trainable Parameters": trainable_params,
        "Frozen Parameters": frozen_params,
        "Input Size": f"3x{Config.IMAGE_SIZE[0]}x{Config.IMAGE_SIZE[1]}",
        "Output Classes": Config.NUM_CLASSES,
        "Device": Config.DEVICE
    }
    
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    out_path = os.path.join(Config.RESULTS_PATH, "model_summary.json")
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    return summary

if __name__ == "__main__":
    # Test instantiation
    model = create_model()
    print("Model created successfully.")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
