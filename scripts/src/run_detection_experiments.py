import os
import sys
import itertools
import torch
import torch.backends.cudnn as cudnn

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.data_loaders import create_dataloaders
from utils.training_utils import train_evaluate_test_detection_model
from utils.architectures import get_detection_model

# Enable CUDA optimisations
cudnn.benchmark = True

# Dataset paths
DATA_DIR_FILE = "./data/File5"
EXPERIMENT_DIR = "./results/detection/STP3_detection_tests"

def run_detection_experiments():
    """Runs object detection experiments with various model architectures.
    
    Tests models:
    - RetinaNet with ResNet50 backbone
    - Faster R-CNN with ResNet50 backbone
    - MobileNetV3-Large with FPN
    
    Not doing transformations as they are already applied to the tags.
    Not rotating because then then the tag location would have to be recalculated and nobody wants to do that just for fun.
    Results are saved in separate directories for each configuration.
    """
    models = ["RetinaNet-ResNet50", "FasterRCNN-ResNet50", "MobileNetV3-Large-FPN"]
    batch_sizes = [4]  # Limited by GPU memory for detection models

    # Test each model configuration
    for model_name, batch_size in itertools.product(models, batch_sizes):
        print(f"Starting detection experiment: Model={model_name}, Batch Size={batch_size}")

        # Set up result directory structure
        result_dir = os.path.join(EXPERIMENT_DIR, model_name, f"batch_size_{batch_size}")
        os.makedirs(result_dir, exist_ok=True)

        # Initialise model on appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_detection_model(model_name).to(device)

        # Create data loaders for detection task
        train_loader, val_loader, test_loader = create_dataloaders(
            root_dir=DATA_DIR_FILE,
            task='detection',
            batch_size=batch_size,
            num_workers=2, 
            train_split=0.8,
            val_split=0.1,
            shuffle=True, 
        )

        # Uncomment to verify detection data format
        # for images, targets, _ in train_loader:
        #     print(f"Image tensor shape: {images[0].shape}")  # First image in batch
        #     print(f"First target: {targets[0]}")  # First target in batch
        #     print(f"First target boxes: {targets[0]['boxes']}")
        #     break  

        # Train and evaluate detection model
        train_evaluate_test_detection_model(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=50,  
            lr=1e-4,
            results_dir=result_dir,
            models_dir=result_dir,
            early_stopping_threshold=99.0
        )

        print(f"Completed detection experiment: Model={model_name}, Batch Size={batch_size}\n")

if __name__ == "__main__":
    run_detection_experiments()
