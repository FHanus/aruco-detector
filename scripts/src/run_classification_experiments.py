import os
import sys
import itertools
import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import v2

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.data_loaders import create_dataloaders
from utils.training_utils import train_evaluate_test_model
from utils.architectures import get_model
from utils.config import get_script_config

# Enable CUDA optimisations
cudnn.benchmark = True

# Load configuration
config = get_script_config('run_classification_experiments')

def get_transforms(transform_name):
    """Returns data augmentation transforms for training.
    
    Available transforms:
    - random_rotation: 360-degree rotations
    - random_blur: Gaussian blur with kernel size 5
    - random_noise: Gaussian noise
    - rotation_blur_noise: Combined transforms
    """
    if transform_name == "random_rotation":
        return v2.Compose([
            v2.RandomRotation(360),
        ])
    elif transform_name == "random_blur":
        return v2.Compose([
            v2.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.5),
        ])
    elif transform_name == "random_noise":
        return v2.Compose([
            v2.RandomApply([v2.GaussianNoise(mean=0.0)], p=0.5),
        ])
    elif transform_name == "rotation_blur_noise":
        return v2.Compose([
            v2.RandomRotation(360),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.5),
            v2.RandomApply([v2.GaussianNoise(mean=0.0)], p=0.5),
        ])
    else:
        return None

def run_experiments():
    """Runs classification experiments with different model architectures and configurations.
    
    Tests combinations of:
    - Models: MinimalCNN, AlexNet variants, ResNet18, GoogLeNet
    - Batch sizes: 32, 64
    - Data augmentations: none, random_rotation, rotation_blur_noise
    
    Results are saved in separate directories for each configuration.
    """
    data = config['experiments']['datasets']
    models = config['experiments']['models']
    batch_sizes = config['experiments']['batch_sizes']
    transformations = config['experiments']['transformations']

    # Test all combinations of parameters
    for training_data, model_name, batch_size, transform_name in itertools.product(data, models, batch_sizes, transformations):
        print(f"Starting experiment: Dataset={training_data}, Model={model_name}, Batch Size={batch_size}, Transformation={transform_name}")

        # Extract the dataset name from the path (second to last directory)
        dataset_name = os.path.normpath(training_data).split(os.sep)[1]

        # Set up result directory structure
        result_dir = os.path.join(config['paths']['EXPERIMENT_DIR'], dataset_name, model_name, f"batch_size_{batch_size}", f"transforms_{transform_name}")
        os.makedirs(result_dir, exist_ok=True)

        # Initialise model and transforms
        model = get_model(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
        transform = get_transforms(transform_name)

        # Create data loaders with specified splits
        train_loader, val_loader, test_loader = create_dataloaders(
            root_dir=training_data,
            task='classification',
            batch_size=batch_size,
            num_workers=config['experiments']['num_workers'],
            train_split=config['experiments']['train_split'],
            val_split=config['experiments']['val_split'],
            shuffle=config['experiments']['shuffle'], 
            transform=transform
        )

        # Verify data transformations
        for images, _, _ in train_loader:
            print(f"Image tensor shape: {images.shape}")
            print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
            break  

        # Train and evaluate model
        train_evaluate_test_model(
            model=model,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=config['experiments']['num_epochs'],
            lr=config['experiments']['lr'],
            results_dir=result_dir,
            models_dir=result_dir,
            early_stopping_threshold=config['experiments']['early_stopping_threshold']
        )

        print(f"Completed experiment: Dataset={training_data}, Model={model_name}, Batch Size={batch_size}, Transformation={transform_name}\n")

if __name__ == "__main__":
    run_experiments()
