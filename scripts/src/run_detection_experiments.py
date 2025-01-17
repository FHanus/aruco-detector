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
from utils.config import get_script_config

# Enable CUDA optimisations
cudnn.benchmark = True

# Load configuration
config = get_script_config('run_detection_experiments')

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
    data = config['experiments']['datasets']
    models = config['experiments']['models']
    batch_sizes = config['experiments']['batch_sizes']

    # Test each model configuration
    for training_data, model_name, batch_size in itertools.product(data, models, batch_sizes):
        print(f"Starting detection experiment: Dataset={training_data}, Model={model_name}, Batch Size={batch_size}")

        # Extract the dataset name from the path (second to last directory)
        dataset_name = os.path.normpath(training_data).split(os.sep)[1]

        # Set up result directory structure
        result_dir = os.path.join(config['paths']['EXPERIMENT_DIR'], dataset_name, model_name, f"batch_size_{batch_size}")
        os.makedirs(result_dir, exist_ok=True)

        # Initialise model on appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_detection_model(model_name).to(device)

        # Create data loaders for detection task
        train_loader, val_loader, test_loader = create_dataloaders(
            root_dir=training_data,
            task='detection',
            batch_size=batch_size,
            num_workers=config['experiments']['num_workers'], 
            train_split=config['experiments']['train_split'],
            val_split=config['experiments']['val_split'],
            shuffle=config['experiments']['shuffle']
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
            num_epochs=config['experiments']['num_epochs'],  
            lr=config['experiments']['lr'],
            results_dir=result_dir,
            models_dir=result_dir,
            early_stopping_threshold=config['experiments']['early_stopping_threshold']
        )

        print(f"Completed detection experiment: Dataset={training_data}, Model={model_name}, Batch Size={batch_size}\n")

if __name__ == "__main__":
    run_detection_experiments()
