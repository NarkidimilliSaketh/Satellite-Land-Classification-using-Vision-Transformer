import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import datasets
from collections import Counter
from PIL import Image
from fpdf import FPDF
from config import Config

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class DatasetExplorer:
    def __init__(self):
        self.dataset_path = Config.DATASET_PATH
        self.analysis_dir = os.path.join(Config.RESULTS_PATH, "dataset_analysis")
        ensure_dir(self.analysis_dir)
        self.dataset = datasets.EuroSAT(root=self.dataset_path, download=True)
        self.classes = Config.CLASS_NAMES
        
    def generate_class_distribution(self):
        print("Generating class distribution plot...")
        targets = [t for _, t in self.dataset]
        counts = Counter(targets)
        
        # Sort classes by index
        labels = [self.classes[i] for i in range(len(self.classes))]
        values = [counts[i] for i in range(len(self.classes))]
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=labels, y=values, palette="viridis")
        plt.title("Class Distribution in EuroSAT Dataset")
        plt.xlabel("Class Name")
        plt.ylabel("Number of Images")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        dist_path = os.path.join(self.analysis_dir, "class_distribution.png")
        plt.savefig(dist_path, dpi=300)
        plt.close()
        return dist_path, counts

    def generate_sample_grid(self):
        print("Generating sample grid...")
        samples = {}
        # Find one sample per class
        for img, label in self.dataset:
            if label not in samples:
                samples[label] = img
            if len(samples) == len(self.classes):
                break
                
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for idx in range(len(self.classes)):
            axes[idx].imshow(samples[idx])
            axes[idx].set_title(self.classes[idx])
            axes[idx].axis('off')
            
        plt.tight_layout()
        grid_path = os.path.join(self.analysis_dir, "sample_grid.png")
        plt.savefig(grid_path, dpi=300)
        plt.close()
        return grid_path
        
    def generate_resolution_distribution(self):
        print("Generating resolution distribution...")
        # Since EuroSAT is 64x64 everywhere, we'll sample the first 500 images to verify
        # to save time instead of all 27000.
        widths, heights = [], []
        for i in range(min(500, len(self.dataset))):
            img, _ = self.dataset[i]
            w, h = img.size
            widths.append(w)
            heights.append(h)
            
        plt.figure(figsize=(8, 6))
        plt.scatter(widths, heights, alpha=0.5, color='blue')
        plt.title("Image Resolution Distribution (Sample)")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.grid(True)
        plt.tight_layout()
        
        res_path = os.path.join(self.analysis_dir, "resolution_distribution.png")
        plt.savefig(res_path, dpi=300)
        plt.close()
        return res_path
        
    def generate_class_information(self, counts):
        print("Generating class_information.json...")
        info = []
        descriptions = {
            "AnnualCrop": "Land used for crop cultivation that is harvested annually.",
            "Forest": "Areas covered with dense trees and vegetation.",
            "HerbaceousVegetation": "Areas dominated by non-woody plants.",
            "Highway": "Major paved roads and highways.",
            "Industrial": "Industrial zones, factories, and commercial structures.",
            "Pasture": "Land used for grazing animals.",
            "PermanentCrop": "Land used for crops that are not replanted annually (e.g. orchards).",
            "Residential": "Housing areas and neighborhoods.",
            "River": "Natural flowing water bodies.",
            "SeaLake": "Large bodies of standing water like lakes and seas."
        }
        
        for idx in range(len(self.classes)):
            class_name = self.classes[idx]
            info.append({
                "Class Name": class_name,
                "Class Index": idx,
                "Number of Images": counts[idx],
                "Description": descriptions.get(class_name, "N/A")
            })
            
        info_path = os.path.join(self.analysis_dir, "class_information.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        return info

    def generate_pdf_report(self, counts, dist_path, grid_path):
        print("Generating dataset_report.pdf...")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Dataset Analysis Report: EuroSAT", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Dataset Summary", ln=True)
        pdf.set_font("Arial", '', 11)
        
        total_images = len(self.dataset)
        train_size = int(0.7 * total_images)
        val_size = int(0.15 * total_images)
        test_size = total_images - train_size - val_size
        
        pdf.cell(0, 8, f"Total Images: {total_images}", ln=True)
        pdf.cell(0, 8, f"Number of Classes: {len(self.classes)}", ln=True)
        pdf.cell(0, 8, f"Training Split (70%): {train_size} images", ln=True)
        pdf.cell(0, 8, f"Validation Split (15%): {val_size} images", ln=True)
        pdf.cell(0, 8, f"Test Split (15%): {test_size} images", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Images per Class:", ln=True)
        pdf.set_font("Arial", '', 11)
        for idx in range(len(self.classes)):
            pdf.cell(0, 8, f"- {self.classes[idx]}: {counts[idx]}", ln=True)
            
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Class Distribution", ln=True)
        pdf.image(dist_path, w=180)
        
        pdf.ln(10)
        pdf.cell(0, 10, "Sample Images (One per Class)", ln=True)
        pdf.image(grid_path, w=180)
        
        report_path = os.path.join(self.analysis_dir, "dataset_report.pdf")
        pdf.output(report_path)
        print(f"Report saved to {report_path}")

    def run(self):
        dist_path, counts = self.generate_class_distribution()
        grid_path = self.generate_sample_grid()
        res_path = self.generate_resolution_distribution()
        self.generate_class_information(counts)
        self.generate_pdf_report(counts, dist_path, grid_path)
        print("Dataset exploration complete!")

if __name__ == "__main__":
    explorer = DatasetExplorer()
    explorer.run()
