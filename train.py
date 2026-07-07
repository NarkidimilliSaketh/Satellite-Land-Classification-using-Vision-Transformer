import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import Config
from dataset import load_data
from model import create_model, unfreeze_backbone
from utils import EarlyStopping, save_checkpoint, load_checkpoint, HistoryTracker
from metrics import AverageMeter, calculate_accuracy, MetricLogger
from visualize import plot_training_curves, generate_training_report

def train_epoch(model, dataloader, criterion, optimizer, scaler, epoch, writer, dry_run=False):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    device = Config.DEVICE
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        acc = calculate_accuracy(outputs, labels)
        
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
        
        # Log batch to Tensorboard
        step = (epoch - 1) * len(dataloader) + batch_idx
        writer.add_scalar('Batch/Train_Loss', loss.item(), step)
        writer.add_scalar('Batch/Train_Acc', acc, step)
        
        if dry_run and batch_idx >= 1:
            break
        
    return loss_meter.avg, acc_meter.avg

def validate(model, dataloader, criterion, dry_run=False):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    device = Config.DEVICE
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            acc = calculate_accuracy(outputs, labels)
            
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))
            
            if dry_run and batch_idx >= 1:
                break
            
    return loss_meter.avg, acc_meter.avg

def main():
    parser = argparse.ArgumentParser(description="Train ViT for Land Use Classification")
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--unfreeze', action='store_true', help='Unfreeze the entire backbone')
    parser.add_argument('--dry-run', action='store_true', help='Run for 1 epoch only for testing')
    parser.add_argument('--fast-train', action='store_true', help='Use 10% of dataset for faster CPU training')
    args = parser.parse_args()

    Config.set_seed()
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    subset_ratio = 0.1 if args.fast_train else 1.0
    train_loader, val_loader, _ = load_data(subset_ratio=subset_ratio)
    
    model = create_model(freeze_backbone=not args.unfreeze).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = torch.amp.GradScaler() if device == "cuda" else None
    
    start_epoch = 1
    if args.resume:
        checkpoint_path = os.path.join(Config.MODEL_PATH, 'last_model.pth')
        start_epoch, _, _ = load_checkpoint(model, optimizer, checkpoint_path)
        start_epoch += 1
        
    tracker = HistoryTracker()
    early_stopping = EarlyStopping(patience=5)
    logger = MetricLogger(total_epochs=Config.EPOCHS)
    writer = SummaryWriter(log_dir=os.path.join(Config.LOGS_PATH, 'tensorboard'))
    
    # Save initial config
    Config.save_config(os.path.join(Config.MODEL_PATH, 'config.json'))

    for epoch in range(start_epoch, Config.EPOCHS + 1):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, writer, args.dry_run)
        val_loss, val_acc = validate(model, val_loader, criterion, args.dry_run)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        logger.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time)
        tracker.update(epoch, train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time)
        
        # Tensorboard Logging
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
        writer.add_scalar('Epoch/Train_Acc', train_acc, epoch)
        writer.add_scalar('Epoch/Val_Acc', val_acc, epoch)
        writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
        
        # Early Stopping check
        early_stopping(val_loss)
        is_best = early_stopping.best_loss == val_loss
        
        save_checkpoint(model, optimizer, epoch, val_loss, val_acc, is_best)
        
        if args.dry_run:
            print("Dry run complete.")
            break
            
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    writer.close()
    
    # Generate post-training visualizations and report
    print("Generating visualizations and reports...")
    plot_training_curves()
    generate_training_report()
    print("Training pipeline finished.")

if __name__ == '__main__':
    main()
