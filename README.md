# Satellite Land Use Classification using Vision Transformer (ViT)

This is a production-ready deep learning application that classifies satellite images into multiple land-use categories using a pretrained Vision Transformer (ViT). 

The primary dataset used is EuroSAT.

## Features
- **Transfer Learning with ViT**: Leverages the pretrained ViT-Base (ViT-B/16) model for powerful image representations.
- **Reproducibility**: Global seed management for consistent results.
- **Automated Data Pipeline**: Automatic downloading, splitting, and processing of the EuroSAT dataset.
- **Production-Ready Structure**: Clean software engineering practices with a modular layout for research and deployment.

## Folder Structure
```text
Satellite-LandUse-Classification/
├── assets/                     # Media, images, and other static assets
├── dataset/                    # Downloaded EuroSAT dataset
├── docs/                       # Project documentation
├── logs/
│   └── tensorboard/            # Tensorboard training logs
├── models/
│   ├── checkpoints/            # Model weight checkpoints saved during training
│   └── pretrained/             # Pretrained base models
├── outputs/
│   ├── predictions/            # Evaluation output predictions
│   └── gradcam/                # Grad-CAM visualization outputs
├── results/
│   ├── confusion_matrix/       # Generated confusion matrices
│   └── reports/                # Summary reports (e.g., dataset_summary.json)
├── tests/                      # Unit tests
├── config.py                   # Centralized configuration settings
├── dataset.py                  # Data loading, transforms, and splits
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Setup & Installation

1. **Clone the repository** and navigate to the project directory.

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The EuroSAT dataset is automatically downloaded and split into training, validation, and testing subsets when `dataset.py` is executed. A summary report is generated and saved as `dataset_summary.json` in the `results/reports/` directory.

To verify data loading and check the summary:
```bash
python dataset.py
```
<img width="1891" height="706" alt="image" src="https://github.com/user-attachments/assets/f59c7581-f8bd-4a03-ace7-94f444e8a206" />
<img width="1914" height="901" alt="image" src="https://github.com/user-attachments/assets/d24f29b6-1b57-4f3a-8681-8559a47c4898" />
<img width="1673" height="565" alt="image" src="https://github.com/user-attachments/assets/3fada1df-4a13-42d0-92ba-3feb67870f9d" />


