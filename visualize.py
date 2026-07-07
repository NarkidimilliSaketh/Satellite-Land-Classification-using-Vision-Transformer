import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from config import Config
from PIL import Image

def plot_training_curves():
    """Generates accuracy, loss, learning rate curves and a combined dashboard."""
    history_file = os.path.join(Config.OUTPUT_PATH, 'training_history.csv')
    if not os.path.exists(history_file):
        print("No training history found.")
        return
        
    df = pd.read_csv(history_file)
    epochs = df['Epoch']
    
    # 1. Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['Training Accuracy'], label='Train Accuracy', marker='o')
    plt.plot(epochs, df['Validation Accuracy'], label='Val Accuracy', marker='s')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(Config.RESULTS_PATH, 'accuracy_curve.png')
    plt.savefig(acc_path)
    plt.close()
    
    # 2. Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['Training Loss'], label='Train Loss', marker='o')
    plt.plot(epochs, df['Validation Loss'], label='Val Loss', marker='s')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(Config.RESULTS_PATH, 'loss_curve.png')
    plt.savefig(loss_path)
    plt.close()
    
    # 3. Learning Rate Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['Learning Rate'], label='Learning Rate', color='green', marker='^')
    plt.title('Learning Rate Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    lr_path = os.path.join(Config.RESULTS_PATH, 'learning_rate_curve.png')
    plt.savefig(lr_path)
    plt.close()
    
    # 4. Combined Dashboard
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(epochs, df['Training Accuracy'], label='Train Acc')
    axes[0].plot(epochs, df['Validation Accuracy'], label='Val Acc')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    
    axes[1].plot(epochs, df['Training Loss'], label='Train Loss')
    axes[1].plot(epochs, df['Validation Loss'], label='Val Loss')
    axes[1].set_title('Loss')
    axes[1].legend()
    
    axes[2].plot(epochs, df['Learning Rate'], label='LR', color='green')
    axes[2].set_title('Learning Rate')
    axes[2].set_yscale('log')
    
    dashboard_path = os.path.join(Config.RESULTS_PATH, 'combined_training_dashboard.png')
    plt.tight_layout()
    plt.savefig(dashboard_path)
    plt.close()
    
    return acc_path, loss_path, lr_path, dashboard_path

def generate_training_report():
    """Generates the training_report.pdf using fpdf2."""
    history_file = os.path.join(Config.OUTPUT_PATH, 'training_history.csv')
    if not os.path.exists(history_file):
        return
        
    df = pd.read_csv(history_file)
    best_acc_idx = df['Validation Accuracy'].idxmax()
    best_val_acc = df.loc[best_acc_idx, 'Validation Accuracy']
    best_train_acc = df.loc[best_acc_idx, 'Training Accuracy']
    best_val_loss = df.loc[best_acc_idx, 'Validation Loss']
    total_time = df['Epoch Time'].sum()
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Training Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Project Information", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Name: {Config.PROJECT_NAME}", ln=True)
    pdf.cell(0, 8, f"Model: {Config.MODEL_NAME} (Pretrained IMAGENET1K_V1)", ln=True)
    pdf.cell(0, 8, f"Dataset: EuroSAT ({Config.NUM_CLASSES} classes)", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Hyperparameters", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Batch Size: {Config.BATCH_SIZE}", ln=True)
    pdf.cell(0, 8, f"Max Epochs: {Config.EPOCHS}", ln=True)
    pdf.cell(0, 8, f"Learning Rate: {Config.LEARNING_RATE}", ln=True)
    pdf.cell(0, 8, f"Optimizer: {Config.OPTIMIZER}", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Best Performance (at best validation epoch)", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Best Validation Accuracy: {best_val_acc:.4f}", ln=True)
    pdf.cell(0, 8, f"Associated Training Accuracy: {best_train_acc:.4f}", ln=True)
    pdf.cell(0, 8, f"Validation Loss: {best_val_loss:.4f}", ln=True)
    pdf.cell(0, 8, f"Total Training Time: {total_time/60:.2f} minutes", ln=True)
    
    # Add training curves
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Training Curves", ln=True)
    
    dash_path = os.path.join(Config.RESULTS_PATH, 'combined_training_dashboard.png')
    if os.path.exists(dash_path):
        pdf.image(dash_path, w=190)
    report_path = os.path.join(Config.RESULTS_PATH, 'reports', 'training_report.pdf')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    pdf.output(report_path)
    print(f"Training report generated at {report_path}")

def plot_confusion_matrix(cm, class_names, output_path):
    """Plots and saves the confusion matrix using seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_evaluation_report(metrics_dict, cm_path, output_path):
    """Generates the evaluation_report.pdf."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Evaluation Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Model & Dataset", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Model: {Config.MODEL_NAME}", ln=True)
    pdf.cell(0, 8, f"Dataset: EuroSAT ({Config.NUM_CLASSES} classes)", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Overall Performance", ln=True)
    pdf.set_font("Arial", '', 11)
    
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            pdf.cell(0, 8, f"{k}: {v:.4f}", ln=True)
        else:
            pdf.cell(0, 8, f"{k}: {v}", ln=True)
            
    if os.path.exists(cm_path):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Confusion Matrix", ln=True)
        pdf.image(cm_path, w=180)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)

def reshape_transform(tensor, height=14, width=14):
    """Reshape transform for ViT in pytorch-grad-cam."""
    # ViT usually outputs [batch_size, num_patches + 1, hidden_dim]
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def generate_gradcam(model, input_tensor, original_image, target_category, output_path):
    """Generates and saves Grad-CAM heatmap overlay for ViT."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("grad-cam not installed. Cannot generate Grad-CAM.")
        return
        
    # For torchvision ViT, the target layer is usually the last encoder block's layer norm
    target_layers = [model.encoder.layers[-1].ln_1]
    
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    # Original image is expected to be [0, 1] numpy float32
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image).astype(np.float32) / 255.0
        # Resize to match Config.IMAGE_SIZE if needed
        if original_image.size != Config.IMAGE_SIZE:
            img_np = np.array(original_image.resize(Config.IMAGE_SIZE)).astype(np.float32) / 255.0
    else:
        img_np = original_image
        
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title('Attention Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(visualization)
    axes[2].set_title(f'Overlay (Target Class ID: {target_category})')
    axes[2].axis('off')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    plot_training_curves()
    generate_training_report()
