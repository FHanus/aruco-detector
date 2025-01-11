import os
import itertools
import torch
from torchvision import transforms as T, models
from torchvision.transforms import v2
from utils.data_loaders import create_dataloaders
from utils.architectures import MinimalCNN
from utils.training_utils import train_evaluate_test_model

# Models to test
def get_model(model_name, num_classes=100):
    if model_name == "MinimalCNN":
        return MinimalCNN(num_classes=num_classes)
    elif model_name == "AlexNet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Transforms to test
def get_transforms(transform_name):
    if transform_name == "random_rotation":
        return v2.Compose([
            v2.RandomRotation(360),
            v2.ToDtype(torch.float32, scale=False)
        ])
    elif transform_name == "random_blur":
        return v2.Compose([
            v2.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.5),
            v2.ToDtype(torch.float32, scale=False)
        ])
    elif transform_name == "random_noise":
        return v2.Compose([
            v2.ToDtype(torch.float32, scale=False),
            v2.RandomApply([v2.GaussianNoise(mean=0.0)], p=0.5)
        ])
    else:
        return v2.Compose([
            v2.ToDtype(torch.float32, scale=False)
        ])

def run_experiments():
    models = ["MinimalCNN", "AlexNet", "ResNet18"]
    batch_sizes = [32, 64, 128]
    transformations = ["random_rotation", "random_blur", "random_noise"]

    DATA_DIR_FILE = "./data/FileCustom1/arucoCombinedDif"

    # Iterate through all the possible combinations
    for model_name, batch_size, transform_name in itertools.product(models, batch_sizes, transformations):
        print(f"Starting experiment: Model={model_name}, Batch Size={batch_size}, Transformation={transform_name}")
        
        # Define result directory
        result_dir = os.path.join("results_class_1E", model_name, f"batch_size_{batch_size}", f"transforms_{transform_name}")
        os.makedirs(result_dir, exist_ok=True)
        
        model = get_model(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
        transform = get_transforms(transform_name)
        
        train_loader, val_loader, test_loader = create_dataloaders(
            root_dir=DATA_DIR_FILE,
            batch_size=batch_size,
            num_workers=4,
            train_split=0.8,
            val_split=0.1,
            shuffle=True, 
            transform=transform
        )
        
        # TEST
        for images, _, _ in train_loader:
            print(f"Image tensor shape: {images.shape}")
            print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
            break  
        
        train_evaluate_test_model(
            model=model,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=1,
            lr=3e-4,
            results_dir=result_dir,
            models_dir=result_dir
        )
        
        print(f"Completed experiment: Model={model_name}, Batch Size={batch_size}, Transformation={transform_name}\n")

if __name__ == "__main__":
    run_experiments()