import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

def plot_classification_metrics(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    accuracy = accuracy_score(labels, preds)
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average=None,
        zero_division=0 
    )
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    base_path = save_path.rsplit('.', 1)[0]
    
    # Plot settings
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_accuracy))
    
    # 1. Accuracy Plot
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
    
    # 2. Precision Plot
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
    
    # 3. Recall Plot
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
    
    # 4. Confusion matrix plot
    plt.figure(figsize=(12, 10))
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)
    heatmap = sns.heatmap(cm_normalized, annot=False, cmap='viridis')
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Normalized Prediction Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{base_path}_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_progress(history, save_path):
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', marker='o')
    plt.plot(history['val_acc'], label='Validation Accuracy', marker='o')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
