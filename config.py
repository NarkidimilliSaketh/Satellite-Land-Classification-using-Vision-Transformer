import os
import random
import torch
import numpy as np

class Config:
    # Project Metadata
    PROJECT_NAME = "Satellite Land Use Classification using Vision Transformer (ViT)"
    VERSION = "1.0.0"
    AUTHOR = "AI Engineer"

    # Seed
    RANDOM_SEED = 42

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "checkpoints")
    PRETRAINED_PATH = os.path.join(BASE_DIR, "models", "pretrained")
    OUTPUT_PATH = os.path.join(BASE_DIR, "outputs")
    RESULTS_PATH = os.path.join(BASE_DIR, "results")
    LOGS_PATH = os.path.join(BASE_DIR, "logs")

    # Dataset & Classes (EuroSAT default classes)
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
        "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
    ]

    # Model Parameters
    MODEL_NAME = "vit_b_16"
    IMAGE_SIZE = (224, 224)

    # Training Parameters
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingLR"

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 0 # Safe default for Windows multiprocessing

    @classmethod
    def set_seed(cls, seed: int = None):
        """Sets all random seeds for reproducibility."""
        if seed is None:
            seed = cls.RANDOM_SEED
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {seed}")

    @classmethod
    def save_config(cls, path: str):
        """Saves the configuration to a JSON file."""
        import json
        config_dict = {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v) and not isinstance(v, classmethod)
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

