import torch
from utils.data_loaders import create_dataloaders
from utils.architectures import MinimalCNN
from utils.training_utils import train_evaluate_test_model
from run_experiments import get_model

# Reproducibility
SEED = 42
torch.manual_seed(SEED)

DATA_DIR_FILE = "./data/FileCustom2/arucoAugmented"

def train_classifier():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_loader_file, val_loader_file, test_loader_file = create_dataloaders(
        DATA_DIR_FILE,
        batch_size=128,      
        num_workers=4,     
        shuffle=True,       
        train_split=0.95,    
        val_split=0.025    
    )

    # Print dataset sizes
    print("\nDataset Statistics:")
    print(f"Training samples: {len(train_loader_file.dataset)}")
    print(f"Validation samples: {len(val_loader_file.dataset)}")
    print(f"Test samples: {len(test_loader_file.dataset)}")

    # Verify classes are what we expect
    train_labels = [label for _, label, _ in train_loader_file.dataset]
    unique_labels = torch.unique(torch.tensor(train_labels))
    print(f"\nNumber of unique classes: {len(unique_labels)}")
    print(f"Class range: {unique_labels.min().item()} to {unique_labels.max().item()}")

    print("\n=== Training ===")

    custom_model = get_model("ResNet18",num_classes=100).to(device) 
    
    train_evaluate_test_model(
        custom_model, 
        device, 
        train_loader_file, 
        val_loader_file, 
        test_loader_file, 
        num_epochs=500,   
        lr=3e-4,
        results_dir="./results_final_class_500E",
        models_dir="./results_final_class_500E",
        early_stopping_threshold=99.8                   
    )

if __name__ == "__main__":
    train_classifier()