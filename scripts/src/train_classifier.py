import os
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.data_loaders import create_dataloaders
from utils.training_utils import train_evaluate_test_model, validate_one_epoch
from utils.architectures import get_model
from utils.model_analysis import plot_classification_metrics, save_test_predictions
from utils.config import get_script_config

# Enable CUDA optimisations
cudnn.benchmark = True

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)

# Load configuration
config = get_script_config('train_classifier')

def train_classifier():
    """Trains the final classifier model on the custom augmented dataset.
    
    Saves best model and training metrics to results directory.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Create data loaders with specified splits
    train_loader_file, val_loader_file, test_loader_file = create_dataloaders(
        config['paths']['DATA_DIR_FILE'],
        task='classification',
        batch_size=config['training']['batch_size'],      
        num_workers=config['training']['num_workers'],     
        shuffle=config['training']['shuffle'],       
        train_split=config['training']['train_split'],    
        val_split=config['training']['val_split']    
    )

    # Log dataset statistics
    print("\nDataset Statistics:")
    print(f"Training samples: {len(train_loader_file.dataset)}")
    print(f"Validation samples: {len(val_loader_file.dataset)}")
    print(f"Test samples: {len(test_loader_file.dataset)}")

    print("\n=== Training ===")

    # Initialise final model for 100-class classification
    final_model = get_model(config['training']['final_model'],num_classes=100).to(device) 
    
    # Train and evaluate model
    train_evaluate_test_model(
        final_model, 
        device, 
        train_loader_file, 
        val_loader_file, 
        test_loader_file, 
        num_epochs=config['training']['num_epochs'],   
        lr=config['training']['lr'],
        results_dir=os.path.join(config['paths']['EXPERIMENT_DIR'],"training_evaluation"),
        models_dir=os.path.join(config['paths']['EXPERIMENT_DIR'],"training_evaluation"),
        early_stopping_threshold=config['training']['early_stopping_threshold']                   
    )

def evaluate_original_datasets():
    """Evaluates trained model on original basic and challenging datasets.
    
    For each dataset:
    - Loads best model weights
    - Runs evaluation on the original datasets
    - Plots analysis
    - Saves predictions to CSV
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained model
    model = get_model(config['training']['final_model'], num_classes=100).to(device)
    model_path = os.path.join(config['paths']['EXPERIMENT_DIR'], "training_evaluation", f"{config['training']['final_model']}_best.pth")
    model.load_state_dict(torch.load(model_path))
    
    print("Using device:", device)

    print("\n=== Evaluating on Original Datasets: File2 and File3 ===")
    
    original_datasets = config['evaluation']['original_datasets']
    
    # Evaluate on each original dataset
    for dataset_name, dataset_path in original_datasets.items():
        print(f"\nEvaluating on {dataset_name}...")

        # Create data loader for full dataset evaluation
        loader, _, _ = create_dataloaders(
            root_dir=dataset_path,
            batch_size=config['evaluation']['batch_size'],
            num_workers=config['evaluation']['num_workers'],
            train_split=1.0,   # Use entire dataset for evaluation
            val_split=0.0,
            shuffle=False, 
            transform=None
        )
        
        # Run evaluation
        loss, acc, labels, preds, paths = validate_one_epoch(
            model, 
            device, 
            loader, 
            nn.CrossEntropyLoss(), 
            True
        )
        
        print(f"{dataset_name} - Loss: {loss:.4f} | Accuracy: {acc:.4f}%")
        
        # Save evaluation results
        analysis_dir = os.path.join(config['paths']['EXPERIMENT_DIR'], f"{dataset_name}_final_evaluation")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Plot and save metrics
        metrics = plot_classification_metrics(
            labels, 
            preds,
            os.path.join(analysis_dir, f"{dataset_name}_final_analysis.png")
        )
        
        # Save detailed predictions
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
