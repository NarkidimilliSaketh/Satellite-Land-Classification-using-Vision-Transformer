import os
import time
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import Config
from model import create_model
from utils import load_checkpoint
from metrics import get_confidence_level
from visualize import generate_gradcam

class Predictor:
    def __init__(self, use_gradcam=True):
        self.device = Config.DEVICE
        self.classes = Config.CLASS_NAMES
        self.use_gradcam = use_gradcam
        
        self.model = create_model().to(self.device)
        best_model_path = os.path.join(Config.MODEL_PATH, 'best_model.pth')
        if os.path.exists(best_model_path):
            load_checkpoint(self.model, None, best_model_path)
            print("Loaded best_model.pth for inference.")
        else:
            print("Warning: best_model.pth not found. Using untrained weights.")
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def predict_image(self, image_path, save_gradcam=True):
        """Predicts a single image and optionally generates Grad-CAM."""
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
            
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(input_tensor)
            else:
                outputs = self.model(input_tensor)
        inf_time = time.time() - start_time
        
        probs = F.softmax(outputs, dim=1)[0]
        
        # Get top-5 predictions
        top5_probs, top5_indices = torch.topk(probs, 5)
        
        top1_idx = top5_indices[0].item()
        top1_prob = top5_probs[0].item()
        pred_class = self.classes[top1_idx]
        conf_level = get_confidence_level(top1_prob)
        
        top5_results = []
        for i in range(5):
            idx = top5_indices[i].item()
            top5_results.append({
                "Class": self.classes[idx],
                "Probability": top5_probs[i].item()
            })
            
        result = {
            "Image": image_path,
            "Predicted Class": pred_class,
            "Confidence": top1_prob,
            "Confidence Level": conf_level,
            "Top-5 Predictions": top5_results,
            "Inference Time (s)": inf_time
        }
        
        if self.use_gradcam and save_gradcam:
            out_name = os.path.basename(image_path).split('.')[0] + "_gradcam.png"
            gradcam_path = os.path.join(Config.OUTPUT_PATH, 'gradcam', out_name)
            
            # Re-enable gradients just for gradcam, model still in eval mode
            for param in self.model.parameters():
                param.requires_grad = True
                
            generate_gradcam(self.model, input_tensor, image, top1_idx, gradcam_path)
            
            # Disable again
            for param in self.model.parameters():
                param.requires_grad = False
                
            result["Grad-CAM Path"] = gradcam_path
            
        return result
        
    def predict_batch(self, directory):
        """Predicts all images in a directory."""
        results = []
        valid_exts = ['.jpg', '.jpeg', '.png']
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in valid_exts):
                filepath = os.path.join(directory, file)
                res = self.predict_image(filepath, save_gradcam=False)
                if res:
                    results.append(res)
        return results

def main():
    parser = argparse.ArgumentParser(description="Predict Land Use Class from Satellite Image")
    parser.add_argument('--image', type=str, help='Path to a single image for prediction')
    parser.add_argument('--dir', type=str, help='Path to a directory of images for batch prediction')
    parser.add_argument('--no-gradcam', action='store_true', help='Disable Grad-CAM generation')
    args = parser.parse_args()
    
    if not args.image and not args.dir:
        print("Please provide an --image or --dir argument.")
        return
        
    predictor = Predictor(use_gradcam=not args.no_gradcam)
    
    if args.image:
        res = predictor.predict_image(args.image)
        if res:
            print("\nPrediction Results:")
            print(f"Image: {res['Image']}")
            print(f"Predicted Class: {res['Predicted Class']} ({res['Confidence Level']})")
            print(f"Confidence: {res['Confidence']:.4f}")
            print(f"Inference Time: {res['Inference Time (s)']:.4f}s")
            print("\nTop-5 Predictions:")
            for p in res['Top-5 Predictions']:
                print(f"- {p['Class']}: {p['Probability']:.4f}")
            if 'Grad-CAM Path' in res:
                print(f"\nGrad-CAM saved to: {res['Grad-CAM Path']}")
                
    if args.dir:
        print(f"\nRunning batch prediction on {args.dir}...")
        results = predictor.predict_batch(args.dir)
        print(f"Processed {len(results)} images.")
        
if __name__ == "__main__":
    main()
