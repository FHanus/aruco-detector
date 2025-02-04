import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torchvision.ops as ops

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.data_loaders import create_dataloaders
from utils.training_utils import train_evaluate_test_detection_model, validate_one_epoch_detection
from utils.architectures import get_detection_model
from utils.model_analysis import plot_detection_metrics, save_detection_predictions
from utils.config import get_script_config

# Enable CUDA optimisations
cudnn.benchmark = True

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)

# Load configuration
config = get_script_config('train_detector')

def train_detector():
    """Trains the final detector model on the custom dataset.
    
    Saves best model and training metrics to results directory.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Create data loaders with specified splits
    train_loader_file, val_loader_file, test_loader_file = create_dataloaders(
        config['paths']['DATA_DIR_FILE'],
        task='detection',
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

    # Initialize RetinaNet model
    final_model = get_detection_model(config['training']['final_model']).to(device)
    
    # Train and evaluate model
    train_evaluate_test_detection_model(
        final_model, 
        device, 
        train_loader_file, 
        val_loader_file, 
        test_loader_file, 
        num_epochs=config['training']['num_epochs'],   
        lr=config['training']['lr'],
        results_dir=os.path.join(config['paths']['EXPERIMENT_DIR'], "training_evaluation"),
        models_dir=os.path.join(config['paths']['EXPERIMENT_DIR'], "training_evaluation"),
        early_stopping_threshold=config['training']['early_stopping_threshold']                   
    )

def evaluate_test_datasets():
    """Evaluates trained model on the original File4 and File5 test datasets.
    
    For each dataset:
    - Loads best model weights
    - Runs evaluation
    - Plots analysis data
    - Saves predictions to CSV
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained model
    model = get_detection_model(config['training']['final_model']).to(device)
    model_path = os.path.join(config['paths']['EXPERIMENT_DIR'], "training_evaluation", f"{model.__class__.__name__}_best.pth")
    print(f"Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path))
    
    print("Using device:", device)

    print("\n=== Evaluating on Test Datasets: File4 and File5 ===")
    
    test_datasets = config['evaluation']['test_datasets']
    
    # Evaluate on each test dataset
    for dataset_name, dataset_path in test_datasets.items():
        print(f"\nEvaluating on {dataset_name}...")

        # Create data loader for full dataset evaluation
        loader, _, _ = create_dataloaders(
            root_dir=dataset_path,
            task='detection',
            batch_size=config['evaluation']['batch_size'],
            num_workers=config['evaluation']['num_workers'],
            train_split=1.0,   # Use entire dataset for evaluation
            val_split=0.0,
            shuffle=False
        )
        
        # Run evaluation
        true_boxes, pred_boxes, paths = validate_one_epoch_detection(
            model, 
            device, 
            loader,
            is_test=True
        )
        
        # Calculate mean IoU
        ious = ops.box_iou(pred_boxes, true_boxes).diagonal()
        mean_iou = ious.mean().item() * 100
        
        print(f"{dataset_name} - Mean IoU: {mean_iou:.2f}%")
        
        # Save evaluation results
        analysis_dir = os.path.join(config['paths']['EXPERIMENT_DIR'], f"{dataset_name}_final_evaluation")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Plot and save metrics
        metrics = plot_detection_metrics(
            pred_boxes,
            true_boxes,
            os.path.join(analysis_dir, f"{dataset_name}_final_analysis.png")
        )
        
        # Save detailed predictions
        save_detection_predictions(
            paths,
            pred_boxes,
            true_boxes,
            metrics,
            os.path.join(analysis_dir, f"{dataset_name}_final_predictions.csv")
        )

if __name__ == "__main__":
    train_detector()
    evaluate_test_datasets()
