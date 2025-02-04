# ArUco Tag Detection and Classification

This project implements deep learning techniques for robust ArUco marker detection and classification, working even in tricky situations like when markers are rotated, blurred or noisy.

## Project Structure

```
.
├── data/                                      
│   ├── File1/                                
│   │   └── arucoRaw/                          # Clean ArUco images (100 images, 512x512)
│   ├── File2/                
│   │   └── arucoBasic/                        # Weakly distorted ArUco images (100 images, 512x512, in each of the 100 class folders)
│   ├── File3/                             
│   │   └── arucoChallenging/                  # Strongly distorted ArUco images (100 images, 512x512, in each of the 100 class folders)
│   ├── File4/       
│   │   └── combinedPicsBasic/                 # Office with randomly placed ArUco (File2) images (100 images, 224x224, and bboxes CSV file)
│   ├── File5/ 
│   │   └── combinedPicsChallenging/           # Office with randomly placed ArUco (File3) images (100 images, 224x224, and bboxes CSV file)
│   ├── File6/                                 
│   │   └── officePics/                        # Original office images (100 images 2560x1440)
│   ├── FileCustom1/                         
│   │   └── arucoAugmented                     # Custom dataset approximating Files 2 and 3
│   └── FileCustom2/                           
│   │   └── combinedAugmented                  # Custom dataset approximating Files 4 and 5
├── scripts/                                   
│   ├── main.py                                # Primary execution script with GPU memory management
│   ├── config/                                # Configuration files for model parameters
│   │   └── config.yaml                        # Main configuration settings
│   ├── src/                                  
│   │   ├── data_exploration.py                # Dataset/results analysis and visualisation
│   │   ├── dataset_augmentation.py            # ArUco augmentation pipeline
│   │   ├── combine_office_tags.py             # Office and ArUco images combination
│   │   ├── run_classification_experiments.py  # Classification testing (STEP 1)
│   │   ├── run_detection_experiments.py       # Detection testing (STEP 3)
│   │   ├── train_classifier.py                # Classification model training (STEP 2)
│   │   └── train_detector.py                  # Detection model training (STEP 4)
│   └── utils/                                
│       ├── architectures.py                   # Model architectures
│       ├── data_loaders.py                    # Dataset loading utilities
│       ├── model_analysis.py                  # Performance analysis tools
│       └── training_utils.py                  # Training helper functions
└── requirements.txt                           
```

## What Each Script Does

### Main Scripts

#### main.py
The primary execution script that manages the entire pipeline:
- Ensures all required data folders exist before processing
- Manages GPU memory between steps (cache clearing and stats reset)
- Provides clear progress updates
- Handles Python path setup for utils module access
- Executes all pipeline scripts in the correct order

#### config.yaml
Main configuration file that stores:
- Model parameters and hyperparameters
- Training settings
- Dataset paths and configurations
- Experiment configurations

#### data_exploration.py
A visualisation tool that lets you look through the datasets and results. You can:
- See stats about the dataset
- Look at where the bounding boxes are, or their predictions (if available)

```bash
python3 data_exploration.py --dataset 
python3 data_exploration.py --detection
```

#### dataset_augmentation.py
Part 1 of the pipeline. Outputs ArUco tag variations by:
- Rotating (0-360°)
- Blurring (σ ranging from 0-14)
- Adding noise (up to 10%<br>of pixels)
- Resizing (50-100%<br>of original)
- The generated dataset is meant for 'train_classifier.py'

#### combine_office_tags.py
Part 2 of the pipeline. Combines ArUco tags with office images:
- Processes office images to 1000x1000 crop
- Resizes to 224x224 final size
- Places 34x34 ArUco tags (from the dataset generated by 'dataset_augmentation.py') at random positions
- Generates bounding box annotations CSV
- The generated dataset is meant for 'train_detector.py'

#### run_classification_experiments.py
Part 3 of the pipeline. Classification model experimentation framework. The only limitation in terms of the amout of tested parameters is the computation time of going through all the iterations:
- Tests multiple architectures 
- Various batch sizes (64, 128)
- Different data augmentation strategies 
- Saves results for each configuration as step 1
- Automatically cleans GPU memory after completion

#### train_classifier.py
Part 4 of the pipeline. Instead of testing the best parameters combination, parameters are selected and the model is trained on the custom augmented (as large as desired) dataset.
- Validates on the originally provided data, that the model has not seen
- Saves results as step 2

#### run_detection_experiments.py
Part 5 of the pipeline. Detection model experimentation framework. The only limitation in terms of the amout of tested parameters is the computation time of going through all the iterations:
- Tests multiple architectures:
  * RetinaNet-ResNet50
  * FasterRCNN-ResNet50
  * MobileNetV3-Large-FPN
- Tiny batch size because I couldn't fit more
- Evaluation metrics
- Results can be visualised using the 'data_exploration.py' script
- Saves results as step 3

#### train_detector.py
Part 6 of the pipeline. Instead of testing the best parameters combination, parameters are selected and the model is trained on the custom augmented (as large as desired) dataset.
- Validates on the originally provided data, that the model has not seen
- Saves results as step 4

### Utility Scripts

#### architectures.py
Model architecture implementations:
- MinimalCNN (custom)
- Pretrained (+ clean) AlexNet 
- Pretrained ResNet18
- Pretrained GoogLeNet
- Detection models (RetinaNet, FasterRCNN, MobileNetV3) - only those that were easily implementable from PyTorch, and worked with no issues out of the box. 

#### data_loaders.py
Dataset handling utilities:
- DataLoaders for all of the training scripts
- Train/val/test split functionality

#### model_analysis.py
Performance analysis tools for both classification and detection:
- Training statistics
- Model performance statistics

#### training_utils.py
Training support functions:
- Training loops
- Validation checks
- Early stopping
- Result logging

## Requirements

- Built with Python 3.10.12
- Matplotlib : Plotting, visualisations
- NumPy : Numerical operations on arrays and matrices
- OpenCV-Python : Image processing functions
- Pandas : DataFrame object handler (CSV)
- Scikit-learn : Tools for model evaluation
- Seaborn : Pretty heatmaps (confusion matrices)
- PyTorch (torch) : ML core

## Getting Started

1. First off, set up your virtual environment:
```bash
python -m venv aruco-venv
source aruco-venv/bin/activate  # Linux/Mac
# or
.\aruco-venv\Scripts\activate   # Windows
```

2. Sort out the dependencies:
```bash
pip install -r requirements.txt
```

## How to Use It

The system's got two main bits:

### Classification

Learns ArUco patterns from Files 2 and 3 (the ones with weak and strong distortion), with additional support for custom datasets in FileCustom1 and FileCustom2. Does this with multiple networks and parameters to compare them. Run:

```bash
python scripts/src/run_classification_experiments.py
```

Results of this are stored as "results/STP1.." 

Targeted script for only training one network with the same purpose can be run with:

```bash
python scripts/src/train_classifier.py
```

Results of this are stored as "results/STP2.." 

### Detection

Locates ArUco patterns (both with weak and strong distortion) in office environment images from Files 4 and 5, with support for custom datasets from FileCustom1 and FileCustom2. Does this with multiple networks and parameters to compare them. Run:

```bash
python scripts/src/run_detection_experiments.py
```

Results of this are stored as "results/STP3.." 

Targeted script for only training one network with the same purpose can be run with:

```bash
python scripts/src/train_detector.py
```

Results of this are stored as "results/STP4.." 

### Full Pipeline

This is the easiest way to run this. Nothing else has to be ran, only this main script to handle it all. To execute the complete pipeline including data preparation, training and evaluation:

```bash
python scripts/main.py
```

Recommended way of running this in order to keep all of the logs, and their time of being outputted is:

```bash
python -u scripts/main.py 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > results/output.log
```

Before running, the script ensures all required data folders exist:
- File1: Contains raw ArUco images (512x512)
- File2: Houses weakly distorted ArUco images
- File3: Stores strongly distorted ArUco images
- File4: Holds office images with randomly placed ArUco tags (basic)
- File5: Contains office images with randomly placed ArUco tags (challenging)
- File6: Keeps the original office images

The script then runs through these steps:
1. Checks and creates any missing required folders
2. Dataset augmentation
3. Office tag combination
4. Classification experiments (results saved as step 1)
5. Classifier training (results saved as step 2)
6. Detection experiments (results saved as step 3)
7. Detector training (results saved as step 4)

The script handles GPU memory cleanup between steps and provides clear progress updates throughout the process.

## Technical Details

### Classification Network

- MinimalCNN: A simple architecture 
  * 3 conv blocks (32->64->128 features)
  * Batch normalisation and dropout
  * Global average pooling
- AlexNet variants:
  * One without any pre-training
  * Another with frozen features from pre-training
- ResNet18 (with weights)
- GoogLeNet (with weights)

Final model:
- GoogLeNet architecture
- Training configuration:
  * Adam optimizer
  * CrossEntropy loss
  * Learning rate: 3e-4
  * Batch size: 32
  * Number of epochs: 50
- Evaluation:
  * Separate testing on File2 and File3
  * Per-class performance analysis
  * Training progress visualisation

### Detection Network

Pre-trained rchitectures:
- RetinaNet with ResNet50 backbone
- Faster R-CNN with ResNet50 backbone
- MobileNetV3-Large with FPN

Common features:
- 224x224 input resolution
- Pretrained ImageNet weights

Final model:
- RetinaNet with ResNet50 backbone (best performing)
- Training configuration:
  * Adam optimizer
  * Learning rate: 1e-4
  * Batch size: 4 (could not fit larger)
  * Number of epochs: 50
- Evaluation:
  * Separate testing on File4 and File5
  * IoU and coordinate-wise metrics
  * Training progress monitoring

## Performance

### Classification Results

#### Step 1

The classification experiments showed varying results across different architectures. It was trained, validated and tested on the File3 dataset.
The final test was performed on a 1000 samples:

| Model Config | Details | Metrics |
|-------------|---------|---------|
| Minimal CNN<br>batch size: 32<br>no transforms | Incorrect predictions: 952 | - Accuracy: 4.80%<br>- Mean precision: 2.60%<br>- Mean recall: 6.07%<br>- Mean F1-score: 2.71% |
| Minimal CNN<br>batch size: 32<br>random rotations | Incorrect predictions: 950 | - Accuracy: 5.00%<br>- Mean precision: 4.11%<br>- Mean recall: 5.53%<br>- Mean F1-score: 3.11% |
| Minimal CNN<br>batch size: 32<br>random rotations, blur and noise | Incorrect predictions: 948 | - Accuracy: 5.20%<br>- Mean precision: 6.92%<br>- Mean recall: 6.85%<br>- Mean F1-score: 3.62% |
| Minimal CNN<br>batch size: 64<br>no transforms | Incorrect predictions: 957 | - Accuracy: 4.30%<br>- Mean precision: 1.86%<br>- Mean recall: 4.33%<br>- Mean F1-score: 2.05% |
| Minimal CNN<br>batch size: 64<br>random rotations | Incorrect predictions: 954 | - Accuracy: 4.60%<br>- Mean precision: 3.00%<br>- Mean recall: 5.63%<br>- Mean F1-score: 3.17% |
| Minimal CNN<br>batch size: 64<br>random rotations, blur and noise | Incorrect predictions: 961 | - Accuracy: 3.90%<br>- Mean precision: 2.32%<br>- Mean recall: 4.61%<br>- Mean F1-score: 2.30% |
| AlexNet (clean)<br>batch size: 32<br>no transforms | Incorrect predictions: 40 | - Accuracy: 96.00%<br>- Mean precision: 95.70%<br>- Mean recall: 96.36%<br>- Mean F1-score: 95.74% |
| AlexNet (clean)<br>batch size: 32<br>random rotations | Incorrect predictions: 29 | - Accuracy: 97.10%<br>- Mean precision: 97.24%<br>- Mean recall: 97.11%<br>- Mean F1-score: 96.99% |
| AlexNet (clean)<br>batch size: 32<br>random rotations, blur and noise | Incorrect predictions: 29 | - Accuracy: 97.10%<br>- Mean precision: 97.24%<br>- Mean recall: 97.11%<br>- Mean F1-score: 96.99% |
| AlexNet (clean)<br>batch size: 64<br>no transforms | Incorrect predictions: 54 | - Accuracy: 94.60%<br>- Mean precision: 95.02%<br>- Mean recall: 94.77%<br>- Mean F1-score: 94.61% |
| AlexNet (clean)<br>batch size: 64<br>random rotations | Incorrect predictions: 14 | - Accuracy: 98.60%<br>- Mean precision: 98.67%<br>- Mean recall: 98.59%<br>- Mean F1-score: 98.51% |
| AlexNet (clean)<br>batch size: 64<br>random rotations, blur and noise | Incorrect predictions: 993 | - Accuracy: 0.70%<br>- Mean precision: 0.01%<br>- Mean recall: 1.00%<br>- Mean F1-score: 0.01% |
| AlexNet<br>batch size: 32<br>no transforms | Incorrect predictions: 283 | - Accuracy: 71.70%<br>- Mean precision: 73.64%<br>- Mean recall: 71.74%<br>- Mean F1-score: 70.49% |
| AlexNet<br>batch size: 32<br>random rotations | Incorrect predictions: 319 | - Accuracy: 68.10%<br>- Mean precision: 70.55%<br>- Mean recall: 68.59%<br>- Mean F1-score: 67.09% |
| AlexNet<br>batch size: 32<br>random rotations, blur and noise | Incorrect predictions: 293 | - Accuracy: 70.70%<br>- Mean precision: 72.13%<br>- Mean recall: 70.77%<br>- Mean F1-score: 69.49% |
| AlexNet<br>batch size: 64<br>no transforms | Incorrect predictions: 274 | - Accuracy: 72.60%<br>- Mean precision: 75.89%<br>- Mean recall: 73.40%<br>- Mean F1-score: 72.52% |
| AlexNet<br>batch size: 64<br>random rotations | Incorrect predictions: 307 | - Accuracy: 69.30%<br>- Mean precision: 71.13%<br>- Mean recall: 70.16%<br>- Mean F1-score: 68.40% |
| AlexNet<br>batch size: 64<br>random rotations, blur and noise | Incorrect predictions: 263 | - Accuracy: 73.70%<br>- Mean precision: 74.85%<br>- Mean recall: 73.63%<br>- Mean F1-score: 72.15% |
| ResNet18<br>batch size: 32<br>no transforms | Incorrect predictions: 0 | - Accuracy: 100.00%<br>- Mean precision: 100.00%<br>- Mean recall: 100.00%<br>- Mean F1-score: 100.00% |
| ResNet18<br>batch size: 32<br>random rotations | Incorrect predictions: 1 | - Accuracy: 99.90%<br>- Mean precision: 99.93%<br>- Mean recall: 99.83%<br>- Mean F1-score: 99.87% |
| ResNet18<br>batch size: 32<br>random rotations, blur and noise | Incorrect predictions: 10 | - Accuracy: 99.00%<br>- Mean precision: 99.00%<br>- Mean recall: 98.82%<br>- Mean F1-score: 98.79% |
| ResNet18<br>batch size: 64<br>no transforms | Incorrect predictions: 2 | - Accuracy: 99.80%<br>- Mean precision: 99.79%<br>- Mean recall: 99.84%<br>- Mean F1-score: 99.81% |
| ResNet18<br>batch size: 64<br>random rotations | Incorrect predictions: 3 | - Accuracy: 99.70%<br>- Mean precision: 99.71%<br>- Mean recall: 99.72%<br>- Mean F1-score: 99.70% |
| ResNet18<br>batch size: 64<br>random rotations, blur and noise | Incorrect predictions: 5 | - Accuracy: 99.50%<br>- Mean precision: 99.46%<br>- Mean recall: 99.52%<br>- Mean F1-score: 99.45% | 
| GoogLeNet<br>batch size: 32<br>no transforms | Incorrect predictions: 1 | - Accuracy: 99.90%<br>- Mean precision: 99.83%<br>- Mean recall: 99.92%<br>- Mean F1-score: 99.87% |
| GoogLeNet<br>batch size: 32<br>random rotations | Incorrect predictions: 5 | - Accuracy: 99.50%<br>- Mean precision: 99.57%<br>- Mean recall: 99.55%<br>- Mean F1-score: 99.54% |
| GoogLeNet<br>batch size: 32<br>random rotations, blur and noise | Incorrect predictions: 5 | - Accuracy: 99.50%<br>- Mean precision: 99.49%<br>- Mean recall: 99.38%<br>- Mean F1-score: 99.40% | 
| GoogLeNet<br>batch size: 64<br>no transforms | Incorrect predictions: 2 | - Accuracy: 99.80%<br>- Mean precision: 99.76%<br>- Mean recall: 99.85%<br>- Mean F1-score: 99.79% |
| GoogLeNet<br>batch size: 64<br>random rotations | Incorrect predictions: 13 | - Accuracy: 98.70%<br>- Mean precision: 98.56%<br>- Mean recall: 98.86%<br>- Mean F1-score: 98.62% |
| GoogLeNet<br>batch size: 64<br>random rotations, blur and noise | Incorrect predictions: 5 | - Accuracy: 99.50%<br>- Mean precision: 99.60%<br>- Mean recall: 99.34%<br>- Mean F1-score: 99.42% |

#### Step 2

The classification final training showed varying results across different architectures. It was trained, validated and tested on the FileCustom1 dataset:

| Model Config | Details | Metrics |
|-------------|---------|---------|
| GoogLeNet<br>Training Evaluation<br>Total test samples: 2500 | Incorrect predictions: 0 | - Accuracy: 100.00%<br>- Mean precision: 100.00%<br>- Mean recall: 100.00%<br>- Mean F1-score: 100.00% |
| GoogLeNet<br>File 2 Evaluation<br>Total test samples: 10000 | Incorrect predictions: 0 | - Accuracy: 100.00%<br>- Mean precision: 100.00%<br>- Mean recall: 100.00%<br>- Mean F1-score: 100.00% |
| GoogLeNet<br>File 3 Evaluation<br>Total test samples: 10000 | Incorrect predictions: 252 | - Accuracy: 97.48%<br>- Mean precision: 97.68%<br>- Mean recall: 97.48%<br>- Mean F1-score: 97.51% |

### Detection Results

#### Step 3

The detection experiments were limited compared to the classification experiments. The code is ready for easy implementation of a similar set of experiments, if a GPU with more VRAM would be used.
The experiments showed varying results across different architectures. It was trained, validated and tested on the File5 dataset.
The final test was performed on a 1000 samples:

| Model Config | Details | Metrics |
|-------------|---------|---------|
| RetinaNet-ResNet50 | batch size: 4 | - Mean IoU: 98.05%<br>- Mean MAE: 98.25%<br>- IoUs over 50%: 100.00% |
| FasterRCNN-ResNet50 | batch size: 4 | - Mean IoU: 97.80%<br>- Mean MAE: 97.60%<br>- IoUs over 50%: 100.00% |
| MobileNetV3-Large-FPN | batch size: 4 | - Mean IoU: 95.93%<br>- Mean MAE: 96.32%<br>- IoUs over 50%: 100.00% |

#### Step 4

The classification final training showed varying results across different architectures. It was trained, validated and tested on the FileCustom1 dataset:

| Model Config | Details | Metrics |
|-------------|---------|---------|
| RetinaNet-ResNet50<br>Training Evaluation | Total test samples: 1000 | - Mean IoU: 99.70%<br>- IoUs over 50%: 100.00% |
| RetinaNet-ResNet50<br>File 4 Evaluation | Total test samples: 100 | - Mean IoU: 89.00%<br>- IoUs over 50%: 100.00% |
| RetinaNet-ResNet50<br>File 5 Evaluation | Total test samples: 100 | - Mean IoU: 85.64%<br>- IoUs over 50%: 97.00% |
