import os
import json
import hashlib
from PIL import Image
from collections import defaultdict
from config import Config

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class DatasetVerifier:
    def __init__(self):
        self.dataset_dir = os.path.join(Config.DATASET_PATH, "eurosat", "2750")
        self.analysis_dir = os.path.join(Config.RESULTS_PATH, "dataset_analysis")
        ensure_dir(self.analysis_dir)
        self.classes = Config.CLASS_NAMES
        
    def hash_file(self, filepath):
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except Exception:
            return None

    def verify(self):
        print("Starting dataset verification...")
        
        missing_classes = []
        empty_folders = []
        corrupted_images = []
        
        # Maps hash to list of filepaths
        hash_dict = defaultdict(list)
        
        total_images_found = 0
        
        if not os.path.exists(self.dataset_dir):
            print(f"Error: Dataset directory {self.dataset_dir} does not exist.")
            # If standard torchvision download didn't extract to eurosat/2750, fallback
            # We'll just look for folders in DATASET_PATH that match class names
            self.dataset_dir = Config.DATASET_PATH
            
        for class_name in self.classes:
            class_dir = os.path.join(self.dataset_dir, class_name)
            
            if not os.path.exists(class_dir):
                missing_classes.append(class_name)
                continue
                
            files = os.listdir(class_dir)
            if len(files) == 0:
                empty_folders.append(class_name)
                continue
                
            for file in files:
                filepath = os.path.join(class_dir, file)
                
                if not os.path.isfile(filepath):
                    continue
                    
                total_images_found += 1
                
                # Check corruption
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                except Exception:
                    corrupted_images.append(filepath)
                    
                # Check duplicate
                file_hash = self.hash_file(filepath)
                if file_hash:
                    hash_dict[file_hash].append(filepath)
                    
        # Find duplicates (hashes with > 1 file)
        duplicates = {k: v for k, v in hash_dict.items() if len(v) > 1}
        num_duplicates = sum(len(v) - 1 for v in duplicates.values())
        
        report = {
            "Total Images Scanned": total_images_found,
            "Missing Classes": missing_classes,
            "Empty Folders": empty_folders,
            "Corrupted Images Found": len(corrupted_images),
            "Corrupted Image Paths": corrupted_images,
            "Exact Duplicates Found (MD5)": num_duplicates,
            "Duplicate Sets": duplicates
        }
        
        report_path = os.path.join(self.analysis_dir, "verification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        print(f"Verification complete. Report saved to {report_path}")
        print(f"Summary: {total_images_found} scanned, {len(corrupted_images)} corrupted, {num_duplicates} duplicates.")

if __name__ == "__main__":
    verifier = DatasetVerifier()
    verifier.verify()
