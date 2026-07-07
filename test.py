import os
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from torchvision.transforms.functional import to_pil_image
from PIL import Image

from config import Config
from model import create_model
from utils import load_checkpoint
from dataset import load_data
from metrics import calculate_topk_accuracy, get_confidence_level
from visualize import plot_confusion_matrix, generate_evaluation_report

def denormalize(tensor):
    """Denormalizes ImageNet values to [0,1] for viewing."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

def save_image(tensor, path):
    img = to_pil_image(denormalize(tensor).cpu())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def test():
    device = Config.DEVICE
    print(f"Starting testing on device: {device}")
    
    _, _, test_loader = load_data()
    model = create_model().to(device)
    
    best_model_path = os.path.join(Config.MODEL_PATH, 'best_model.pth')
    if os.path.exists(best_model_path):
        load_checkpoint(model, None, best_model_path)
        print("Loaded best_model.pth")
    else:
        print("Warning: best_model.pth not found. Evaluating randomly initialized weights.")
        
    model.eval()
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    top1_acc_meter = 0
    top3_acc_meter = 0
    total_samples = 0
    
    inference_times = []
    
    misclassified_dir = os.path.join(Config.RESULTS_PATH, 'misclassified')
    best_preds_dir = os.path.join(Config.RESULTS_PATH, 'best_predictions')
    worst_preds_dir = os.path.join(Config.RESULTS_PATH, 'worst_predictions')
    
    os.makedirs(misclassified_dir, exist_ok=True)
    os.makedirs(best_preds_dir, exist_ok=True)
    os.makedirs(worst_preds_dir, exist_ok=True)
    
    # Store tuples of (confidence, path_to_save, img_tensor, is_correct, pred, target)
    qualitative_results = []
    prediction_records = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            
            start_time = time.time()
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
            else:
                outputs = model(images)
                
            probs = F.softmax(outputs, dim=1)
            inf_time = (time.time() - start_time) / images.size(0)
            inference_times.extend([inf_time]*images.size(0))
            
            top1 = calculate_topk_accuracy(outputs, targets, k=1)
            top3 = calculate_topk_accuracy(outputs, targets, k=3)
            top1_acc_meter += top1 * images.size(0)
            top3_acc_meter += top3 * images.size(0)
            total_samples += images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            for j in range(images.size(0)):
                img = images[j]
                tgt = targets[j].item()
                prd = preds[j].item()
                prob = probs[j][prd].item()
                conf_level = get_confidence_level(prob)
                
                is_correct = (tgt == prd)
                
                record = {
                    "Ground Truth": Config.CLASS_NAMES[tgt],
                    "Predicted Class": Config.CLASS_NAMES[prd],
                    "Confidence": prob,
                    "Confidence Level": conf_level,
                    "Correct": is_correct
                }
                for c_idx in range(Config.NUM_CLASSES):
                    record[f"Prob_{Config.CLASS_NAMES[c_idx]}"] = probs[j][c_idx].item()
                    
                prediction_records.append(record)
                
                # Save misclassified instantly
                if not is_correct:
                    filename = f"batch{i}_img{j}_GT_{Config.CLASS_NAMES[tgt]}_Pred_{Config.CLASS_NAMES[prd]}_Conf_{prob:.2f}.png"
                    save_image(img, os.path.join(misclassified_dir, filename))
                
                qualitative_results.append((prob, img, is_correct, prd, tgt, i, j))
                
    # Qualitative extreme examples saving
    # Best: Correct and highest confidence
    # Worst: Incorrect and highest confidence in the wrong class
    correct_results = sorted([r for r in qualitative_results if r[2]], key=lambda x: x[0], reverse=True)
    incorrect_results = sorted([r for r in qualitative_results if not r[2]], key=lambda x: x[0], reverse=True)
    
    for rank, res in enumerate(correct_results[:10]):
        prob, img, _, prd, tgt, bi, bj = res
        save_image(img, os.path.join(best_preds_dir, f"rank{rank+1}_Conf_{prob:.4f}_{Config.CLASS_NAMES[prd]}.png"))
        
    for rank, res in enumerate(incorrect_results[:10]):
        prob, img, _, prd, tgt, bi, bj = res
        save_image(img, os.path.join(worst_preds_dir, f"rank{rank+1}_Conf_{prob:.4f}_Pred_{Config.CLASS_NAMES[prd]}_GT_{Config.CLASS_NAMES[tgt]}.png"))

    # Compute Metrics
    acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
    top1_acc = top1_acc_meter / total_samples
    top3_acc = top3_acc_meter / total_samples
    avg_inf_time = sum(inference_times) / len(inference_times)
    
    # Reports
    report_dict = classification_report(all_targets, all_preds, target_names=Config.CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    # Exports
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    
    # 1. Prediction CSV
    df = pd.DataFrame(prediction_records)
    df.to_csv(os.path.join(Config.RESULTS_PATH, "prediction.csv"), index=False)
    
    # 2. Classification Report JSON
    with open(os.path.join(Config.RESULTS_PATH, "classification_report.json"), 'w') as f:
        json.dump(report_dict, f, indent=4)
        
    # 3. Metrics JSON & Inference Summary
    metrics = {
        "Accuracy": acc,
        "Top-1 Accuracy": top1_acc,
        "Top-3 Accuracy": top3_acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "Average Inference Time (s)": avg_inf_time
    }
    with open(os.path.join(Config.RESULTS_PATH, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    inference_summary = {
        "Model": Config.MODEL_NAME,
        "Dataset": "EuroSAT",
        "Number of Classes": Config.NUM_CLASSES,
        "Number of Test Images": total_samples,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "Average Inference Time (s)": avg_inf_time
    }
    with open(os.path.join(Config.RESULTS_PATH, "inference_summary.json"), 'w') as f:
        json.dump(inference_summary, f, indent=4)
        
    # 4. Confusion Matrix PNG
    cm_path = os.path.join(Config.RESULTS_PATH, "confusion_matrix.png")
    plot_confusion_matrix(cm, Config.CLASS_NAMES, cm_path)
    
    # 5. Evaluation Report PDF
    pdf_path = os.path.join(Config.RESULTS_PATH, "evaluation_report.pdf")
    generate_evaluation_report(metrics, cm_path, pdf_path)
    
    print("Testing complete. All artifacts saved to results/")

if __name__ == "__main__":
    test()
