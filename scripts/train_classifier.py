import os
import torch
import torch.nn as nn
from utils.data_loaders import create_dataloaders
from utils.architectures import MinimalCNN
from utils.training_utils import train_evaluate_test_model, validate_one_epoch
from run_experiments import get_model
from utils.model_analysis import plot_classification_metrics, plot_training_progress, save_test_predictions

# Reproducibility
SEED = 42
torch.manual_seed(SEED)

DATA_DIR_FILE = "./data/FileCustom2/arucoAugmented"
DATA_ORIGINAL_COMBINED_DIR_FILE = "./data/FileCustom1/arucoCombinedDif"
EXPERIMENT_DIR = "./results_final_class_augmented_ToA"

def train_classifier():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_loader_file, val_loader_file, test_loader_file = create_dataloaders(
        DATA_DIR_FILE,
        task='classification',
        batch_size=128,      
        num_workers=8,     
        shuffle=True,       
        train_split=0.95,    
        val_split=0.025    
    )

    # For validation on provided data (File2 and File3 combined)
    # _, val_loader_file, test_loader_file = create_dataloaders(
    #         root_dir=DATA_ORIGINAL_COMBINED_DIR_FILE,
    #         batch_size=256,
    #         num_workers=4,
    #         train_split=0.0,
    #         val_split=0.5,
    #         shuffle=False, 
    #         transform=None
    # )

    # Print dataset sizes
    print("\nDataset Statistics:")
    print(f"Training samples: {len(train_loader_file.dataset)}")
    print(f"Validation samples: {len(val_loader_file.dataset)}")
    print(f"Test samples: {len(test_loader_file.dataset)}")

    print("\n=== Training ===")

    final_model = get_model("ResNet18",num_classes=100).to(device) 
    
    train_evaluate_test_model(
        final_model, 
        device, 
        train_loader_file, 
        val_loader_file, 
        test_loader_file, 
        num_epochs=100,   
        lr=3e-4,
        results_dir= os.path.join(EXPERIMENT_DIR,"training_evaluation"),
        models_dir=  os.path.join(EXPERIMENT_DIR,"training_evaluation"),
        early_stopping_threshold=99.9                   
    )

def evaluate_original_datasets():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model("ResNet18", num_classes=100).to(device)
    print(os.path.join(EXPERIMENT_DIR,"training_evaluation/ResNet_best.pth"))
    model.load_state_dict(torch.load(os.path.join(EXPERIMENT_DIR,"training_evaluation/ResNet_best.pth")))
    
    print("Using device:", device)

    print("\n=== Evaluating on Original Datasets: File2 and File3 ===")
    
    original_datasets = {
        "File2": "./data/File2/arucoBasic",
        "File3": "./data/File3/arucoChallenging"
    }
    
    for dataset_name, dataset_path in original_datasets.items():
        print(f"\nEvaluating on {dataset_name}...")

        loader, _, _ = create_dataloaders(
            root_dir=dataset_path,
            batch_size=256,
            num_workers=4,
            train_split=1.0,   # Reuse the dataloader for this test
            val_split=0.0,
            shuffle=False, 
            transform=None
        )
        
        # Run validation
        loss, acc, labels, preds, paths = validate_one_epoch(
            model, 
            device, 
            loader, 
            nn.CrossEntropyLoss(), 
            True
        )
        
        print(f"{dataset_name} - Loss: {loss:.4f} | Accuracy: {acc:.4f}%")
        
        # Plot and save metrics
        analysis_dir = os.path.join(EXPERIMENT_DIR, f"{dataset_name}_final_evaluation")
        os.makedirs(analysis_dir, exist_ok=True)
        
        metrics = plot_classification_metrics(
            labels, 
            preds,
            os.path.join(analysis_dir, f"{dataset_name}_final_analysis.png")
        )
        
        # Save predictions to CSV
        save_test_predictions(
            paths,
            labels,
            preds,
            metrics,
            os.path.join(analysis_dir, f"{dataset_name}_final_predictions.csv")
        )

if __name__ == "__main__":
    train_classifier()
    evaluate_original_datasets()