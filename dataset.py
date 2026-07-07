import os
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import Counter
from config import Config

def get_transforms():
    """Returns training and validation transforms for ViT."""
    # ViT typically expects 224x224 images and ImageNet normalization
    train_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def generate_dataset_summary(dataset, train_size, val_size, test_size):
    """Generates and saves a JSON summary of the dataset."""
    
    # Calculate class distribution
    class_counts = Counter([target for _, target in dataset])
    
    # Since torchvision EuroSAT doesn't have an attribute `classes` in older versions natively always,
    # or it might have different formatting, we use our config.
    classes = Config.CLASS_NAMES
    
    class_distribution = {classes[idx]: count for idx, count in class_counts.items()}
    
    # Get image resolution from a sample
    sample_img, _ = dataset[0]
    # sample_img is a PIL image before transforms if we access original dataset
    # But since we might apply transforms later, let's just output the target resolution
    
    summary = {
        "Total Images": len(dataset),
        "Images per Class": class_distribution,
        "Train Count": train_size,
        "Validation Count": val_size,
        "Test Count": test_size,
        "Target Image Resolution": f"{Config.IMAGE_SIZE[0]}x{Config.IMAGE_SIZE[1]}",
        "Class Distribution": class_distribution
    }
    
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    reports_dir = os.path.join(Config.RESULTS_PATH, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    summary_path = os.path.join(reports_dir, "dataset_summary.json")
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Dataset summary saved to {summary_path}")
    return summary

class TransformSubset(torch.utils.data.Dataset):
    """Wrapper to apply transforms dynamically to dataset subsets."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def load_data(subset_ratio=1.0):
    """Downloads (if needed), loads and splits the EuroSAT dataset."""
    
    # Download dataset
    print("Loading EuroSAT Dataset...")
    full_dataset = datasets.EuroSAT(
        root=Config.DATASET_PATH, 
        download=True
    )
    
    # Calculate splits (e.g., 70% Train, 15% Val, 15% Test)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Total dataset size: {total_size}")
    
    # Set seed for reproducibility before splitting
    generator = torch.Generator().manual_seed(Config.RANDOM_SEED)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    if subset_ratio < 1.0:
        print(f"Applying fast training subset ratio: {subset_ratio}")
        train_size = int(train_size * subset_ratio)
        val_size = int(val_size * subset_ratio)
        test_size = int(test_size * subset_ratio)
        
        train_dataset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size], generator=generator)
        val_dataset, _ = random_split(val_dataset, [val_size, len(val_dataset) - val_size], generator=generator)
        test_dataset, _ = random_split(test_dataset, [test_size, len(test_dataset) - test_size], generator=generator)
    
    # Generate summary
    generate_dataset_summary(full_dataset, train_size, val_size, test_size)
    
    # Apply transforms
    train_transform, val_transform = get_transforms()
            
    train_dataset = TransformSubset(train_dataset, transform=train_transform)
    val_dataset = TransformSubset(val_dataset, transform=val_transform)
    test_dataset = TransformSubset(test_dataset, transform=val_transform)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    Config.set_seed()
    train_loader, val_loader, test_loader = load_data()
    print("DataLoaders prepared successfully.")
    
    # Fetch one batch to verify
    images, labels = next(iter(train_loader))
    print(f"Batch Image shape: {images.shape}")
    print(f"Batch Label shape: {labels.shape}")
