import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import torchvision.ops as ops
import torch

def plot_classification_metrics(labels, preds, save_path):
    """Generates and saves classification performance visualisations.
    
    Creates 4 plots:
    - Per-class accuracy
    - Per-class precision
    - Per-class recall
    - Confusion matrix
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Calculate metrics
    cm = confusion_matrix(labels, preds)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average=None,
        zero_division=0 
    )
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    base_path = save_path.rsplit('.', 1)[0]
    
    # Plot per-class accuracy
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_accuracy))
    plt.bar(x, class_accuracy, color='salmon')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title(f'Per-Class Accuracy\nOverall Accuracy: {accuracy:.2%}')
    plt.xticks(x, [f'Class {i}' if i % 2 == 0 else '' for i in range(len(class_accuracy))],
               rotation=90, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_path}_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot per-class precision
    plt.figure(figsize=(12, 6))
    plt.bar(x, precision, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.title(f'Per-Class Precision\nMean Precision: {np.mean(precision):.2%}')
    plt.xticks(x, [f'Class {i}' if i % 2 == 0 else '' for i in range(len(precision))],
               rotation=90, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_path}_precision.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot per-class recall
    plt.figure(figsize=(12, 6))
    plt.bar(x, recall, color='lightgreen')
    plt.xlabel('Class')
    plt.ylabel('Recall')
    plt.title(f'Per-Class Recall\nMean Recall: {np.mean(recall):.2%}')
    plt.xticks(x, [f'Class {i}' if i % 2 == 0 else '' for i in range(len(recall))],
               rotation=90, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_path}_recall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    plt.figure(figsize=(12, 10))
    cm_normalised = cm / cm.sum(axis=1, keepdims=True)
    heatmap = sns.heatmap(cm_normalised, annot=False, cmap='viridis')
    plt.title('Confusion Matrix (Normalised)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Normalised Prediction Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{base_path}_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'accuracy': accuracy,
        'mean_precision': np.mean(precision),
        'mean_recall': np.mean(recall),
        'mean_f1': np.mean(f1),
    }

def plot_detection_metrics(pred_boxes, true_boxes, save_path):
    """Analyses object detection performance and generates visualisations.
    
    Calculates:
    - Mean Absolute Error for box coordinates
    - Intersection over Union statistics
    - Detection accuracy (for IoU > 0.5)
    
    Creates IoU distribution plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base_path = save_path.rsplit('.', 1)[0]

    # Calculate coordinate-wise MAE
    mae = torch.mean(torch.abs(pred_boxes - true_boxes), dim=0)
    mae_total = mae.mean().item()
    
    # Calculate IoU metrics
    ious = ops.box_iou(pred_boxes, true_boxes).diagonal()
    mean_iou = ious.mean().item()
    median_iou = ious.median().item()
    
    # Calculate detection accuracy (IoU threshold 50% to know we are in the correct spot)
    accuracy = (ious > 0.5).float().mean().item()
    
    # Plot IoU distribution
    plt.figure(figsize=(12, 6))
    plt.hist(ious.cpu().numpy(), bins=50, range=(0, 1), color='skyblue', rwidth=0.8)
    plt.xticks(np.arange(0, 1.1, 0.05))
    plt.xlabel('Intersection over Union (IoU)')
    plt.ylabel('Count')
    plt.title('IoU Distribution')
    plt.grid(True, alpha=1)
    plt.tight_layout()
    plt.savefig(f'{base_path}_iou_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'mean_mae': mae_total,
        'mean_iou': mean_iou,
        'median_iou': median_iou,
        'average_precision': accuracy
    }

def plot_training_progress(history, save_path):
    """Plots training metrics over epochs.
    
    Creates dual plots:
    - Loss curve
    - Accuracy/IoU curve (based on task type)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', marker='o')
    if 'train_acc' in history:  # Classification metrics
        plt.plot(history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Accuracy/IoU plot
    plt.subplot(1, 2, 2)
    if 'train_acc' in history:  # Classification metrics
        plt.plot(history['train_acc'], label='Training Accuracy', marker='o')
        plt.plot(history['val_acc'], label='Validation Accuracy', marker='o')
        plt.title('Accuracy vs. Epoch')
        plt.ylabel('Accuracy (%)')
    elif 'val_iou' in history:  # Detection metrics
        plt.plot(history['val_iou'], label='Validation IoU', marker='o')
        plt.title('IoU vs. Epoch')
        plt.ylabel('IoU')
    
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_test_predictions(file_paths, true_labels, predicted_labels, metrics, save_path):    
    """Saves classification results and metrics to CSV.
    
    Includes:
    - File paths
    - True and predicted labels
    - Overall metrics summary
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    results_df = pd.DataFrame({
        'file_path': file_paths,
        'true_class': true_labels,
        'predicted_class': predicted_labels,
        'correct': [t == p for t, p in zip(true_labels, predicted_labels)]
    })
    
    results_df = results_df.sort_values(['correct', 'file_path'])
    
    with open(save_path, 'w') as f:
        f.write(f"Test Set Analysis: {save_path}\n")
        f.write(f"Total samples: {len(results_df)}\n")
        f.write(f"Overall accuracy: {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Mean precision: {metrics['mean_precision']*100:.2f}%\n") 
        f.write(f"Mean recall: {metrics['mean_recall']*100:.2f}%\n")
        f.write(f"Mean F1-score: {metrics['mean_f1']*100:.2f}%\n")
        f.write(f"Incorrect predictions: {len(results_df) - results_df['correct'].sum()}\n\n")
    
    results_df.to_csv(save_path, mode='a', index=False)

def save_detection_predictions(file_paths, pred_boxes, true_boxes, metrics, save_path):
    """Saves detection results and metrics to CSV.
    
    Includes:
    - File paths
    - Predicted and true bounding boxes
    - IoU and MAE metrics
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Format box coordinates as strings
    pred_boxes_str = [f"{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}" for box in pred_boxes]
    true_boxes_str = [f"{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}" for box in true_boxes]
    
    results_df = pd.DataFrame({
        'file_path': file_paths,
        'pred_boxes': pred_boxes_str,
        'true_boxes': true_boxes_str
    })
    
    with open(save_path, 'w') as f:
        f.write(f"Detection Analysis: {save_path}\n")
        f.write(f"Total samples: {len(results_df)}\n")
        f.write(f"Mean IoU: {metrics['mean_iou']:.4f}\n")
        f.write(f"MAE: {metrics['mean_mae']:.4f}\n")
        f.write(f"Accuracy: {metrics['average_precision']:.4f}\n")
    
    results_df.to_csv(save_path, mode='a', index=False)
