import os
import json
import csv
import torch
import shutil
from config import Config

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, is_best):
    """Saves model checkpoints according to requirements."""
    os.makedirs(Config.MODEL_PATH, exist_ok=True)
    epoch_dir = os.path.join(Config.MODEL_PATH, "every_epoch")
    os.makedirs(epoch_dir, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    
    # Save last model
    last_path = os.path.join(Config.MODEL_PATH, 'last_model.pth')
    torch.save(state, last_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(Config.MODEL_PATH, 'best_model.pth')
        shutil.copyfile(last_path, best_path)
        
    # Save every 5 epochs
    if epoch % 5 == 0:
        epoch_path = os.path.join(epoch_dir, f'model_ep{epoch}.pth')
        shutil.copyfile(last_path, epoch_path)
        
    # Save config with checkpoint
    Config.save_config(os.path.join(Config.MODEL_PATH, 'config.json'))

def load_checkpoint(model, optimizer, filepath):
    """Loads a model checkpoint."""
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss'], checkpoint['val_acc']
    else:
        print(f"No checkpoint found at {filepath}")
        return 0, float('inf'), 0.0

class HistoryTracker:
    def __init__(self):
        self.history = []
        self.csv_path = os.path.join(Config.OUTPUT_PATH, 'training_history.csv')
        self.json_path = os.path.join(Config.OUTPUT_PATH, 'training_history.json')
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
        
    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, lr, epoch_time):
        record = {
            "Epoch": epoch,
            "Training Loss": round(train_loss, 4),
            "Validation Loss": round(val_loss, 4),
            "Training Accuracy": round(train_acc, 4),
            "Validation Accuracy": round(val_acc, 4),
            "Learning Rate": lr,
            "Epoch Time": round(epoch_time, 2)
        }
        self.history.append(record)
        self._save_csv()
        self._save_json()
        
    def _save_csv(self):
        if not self.history:
            return
        keys = self.history[0].keys()
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
            
    def _save_json(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.history, f, indent=4)
