# ArUco Detector

This project is part of the **Advanced Vision for Localisation and Mapping** module. It detects and classifies ArUco markers under various conditions, including distortions and noise.

## Folder Structure

The project directory is organised as follows:

```
aruco-detector
├── data/                  # Provided dataset
├── models/                # Directory to save trained models
├── results/               # Directory to store results
├── scripts/               # Python scripts
│ ├── utils/               # Utility modules
│ ├── data_exploration.py  # For viewing dataset images
│ ├── dataset_mixer.py     # For combining datasets
│ ├── run_experiments.py   # For classification experiments
│ └── train_classifier.py  # For a simple training the classifier
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## Dependencies

```bash
pip install -r requirements.txt
```

## Scripts

### Data Exploration
The data explorer and visualiser (`data_exploration.py`) allows you to inspect the dataset images, view histograms, and see bounding box annotations where applicable.

```bash
python scripts/data_exploration.py --dataset [dataset_type]
``` 

Available dataset types:
- `raw`: Raw ArUco marker images
- `basic`: Basic dataset with clean markers
- `challenging`: Challenging dataset with distortions
- `combinedbasic`: Combined dataset with basic markers
- `combinedchallenging`: Combined dataset with challenging markers
- `office`: Office environment dataset
- `custommix`: Folder generated using dataset_mixer.py (see below)

### Dataset Mixing
The dataset mixing script (`dataset_mixer.py`) combines the basic and challenging datasets.

### Model Training 
The classifier training script (`train_classifier.py`) trains a simple CNN model on the ArUco marker dataset. During training it will:

- Save the best performing model
- Generate training progress plots showing loss and accuracy curves
- Create confusion matrix and classification reports
- Save all results to the `results/` directory

The trained model weights will be saved to the `models/` directory.

### Experimental Pipeline
The experiment script (`run_experiments.py`) evaluates different:

- Model architectures (MinimalCNN, AlexNet, ResNet18)
- Batch sizes (32, 64, 128)
- Data augmentations:
  - Random rotation
  - Random blur
  - Random noise

Results for each experiment configuration are saved in organised subdirectories under `results/`.

## Model Architectures

### MinimalCNN
A lightweight CNN architecture designed for ArUco marker classification:
- 3 convolutional layers with batch normalisation
- Global average pooling
- Dropout for regularisation
- Fully connected output layer

### Pre-trained Models
The system also fine-tunes pre-trained models:
- AlexNet
- ResNet18