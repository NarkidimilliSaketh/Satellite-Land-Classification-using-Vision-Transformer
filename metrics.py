import time
import datetime
import torch

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(output, target):
    """Computes the accuracy for the batch."""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum().item()
        acc = correct / target.size(0)
        return acc

class MetricLogger:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.start_time = time.time()
        
    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, lr, epoch_time):
        # Calculate ETA
        elapsed_time = time.time() - self.start_time
        avg_epoch_time = elapsed_time / epoch
        remaining_epochs = self.total_epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        print("-" * 80)
        print(f"Epoch: [{epoch}/{self.total_epochs}] | Time: {epoch_time:.2f}s | ETA: {eta_str}")
        print(f"LR: {lr:.2e}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        print("-" * 80)

def get_confidence_level(prob):
    """Maps a probability [0, 1] to a confidence level."""
    if prob >= 0.90:
        return "Very High"
    elif prob >= 0.75:
        return "High"
    elif prob >= 0.50:
        return "Medium"
    else:
        return "Low"

def calculate_topk_accuracy(output, target, k=3):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = min(k, output.size(1))
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:maxk].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.item() / batch_size

