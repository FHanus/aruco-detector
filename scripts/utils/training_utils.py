import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.ops as ops

from .model_analysis import plot_classification_metrics, plot_training_progress, save_test_predictions, plot_detection_metrics, save_detection_predictions

def train_one_epoch(model, device, train_loader, loss_fn, optimizer):
    """Trains model for one epoch and returns loss and accuracy.
    
    Returns average loss and accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets, _ in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += inputs.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = 100. * total_correct / total_samples

    return epoch_loss, epoch_acc

def validate_one_epoch(model, device, val_loader, loss_fn, is_test=False):
    """Evaluates model on validation/test set.
    
    Returns metrics and optionally file paths for test set evaluation.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_labels = []
    all_preds = []
    all_paths = []

    with torch.no_grad():
        for _, (inputs, targets, paths) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)

            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            if is_test and paths is not None:
                all_paths.extend(paths)

    epoch_loss = total_loss / total_samples
    epoch_acc = 100. * total_correct / total_samples

    if is_test:
        return epoch_loss, epoch_acc, all_labels, all_preds, all_paths
    return epoch_loss, epoch_acc, all_labels, all_preds

def train_evaluate_test_model(model, device, train_loader, val_loader, test_loader,
                       num_epochs=5, lr=1e-3, results_dir="./results", models_dir="./models",
                       early_stopping_threshold=99.0, early_stopping_threshold_val=99.0):    
    """Handles complete training cycle for classification models.
    
    Features:
    - Adam optimiser with CrossEntropy loss
    - Early stopping on accuracy thresholds
    - Best model checkpointing
    - Training progress plots
    - Test set evaluation
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    model_name = model.__class__.__name__
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, device, train_loader, loss_fn, optimizer)
        val_loss, val_acc, val_labels, val_preds = validate_one_epoch(model, device, val_loader, loss_fn)

        end_time = time.time()
        epoch_time = end_time - start_time

        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}% "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}% "
              f"Time: {epoch_time:.2f}s")

        # Save if best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(models_dir, f"{model_name}_best.pth"))

        # Check early stopping criteria
        if val_acc >= early_stopping_threshold_val and train_acc >= early_stopping_threshold:
            print(f"\nReached validation accuracy threshold of {early_stopping_threshold}% at epoch {epoch+1} ({val_acc:.2}%).")
            print("Early stopping triggered.")
            break

    # Generate training visualisations
    plot_training_progress(
        history, 
        os.path.join(results_dir, f"{model_name}_training_progress.png")
    )

    # Evaluate on test set using best model
    model.load_state_dict(torch.load(os.path.join(models_dir, f"{model_name}_best.pth")))
    test_loss, test_acc, test_labels, test_preds, test_paths = validate_one_epoch(
        model, device, test_loader, loss_fn, is_test=True
    )

    print(f"\nTest Set Performance:")
    print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}%")

    # Save analysis plots and predictions
    metrics = plot_classification_metrics(
        test_labels, 
        test_preds,
        os.path.join(results_dir, f"{model_name}_analysis.png")
    )
    
    save_test_predictions(
        test_paths,
        test_labels,
        test_preds,
        metrics,
        os.path.join(results_dir, f"{model_name}_test_predictions.csv")
    )

    return history, (val_labels, val_preds), (test_labels, test_preds)

def train_one_epoch_detection(model, device, train_loader, optimizer):
    """Trains detection model for one epoch.
    
    Returns average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for images, targets, _ in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        total_samples += len(images)

    return total_loss / total_samples

def validate_one_epoch_detection(model, device, val_loader, is_test=False):
    """Evaluates detection model on validation/test set.
    
    Returns true and predicted bounding boxes for analysis.
    """
    model.eval()
    total_samples = 0
    
    all_pred_boxes = []
    all_true_boxes = []
    all_paths = []
    
    with torch.no_grad():
        for images, targets, paths in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)
            
            for pred, target, path in zip(predictions, targets, paths):
                # Select highest confidence prediction or zero box
                if len(pred['boxes']) > 0:
                    best_idx = pred['scores'].argmax()
                    pred_box = pred['boxes'][best_idx].cpu()
                else:
                    pred_box = torch.tensor([0., 0., 0., 0.])
                
                target_box = target['boxes'][0].cpu()
                
                all_pred_boxes.append(pred_box)
                all_true_boxes.append(target_box)
                if is_test and paths is not None:
                    all_paths.append(path)
            
            total_samples += len(images)

    pred_boxes = torch.stack(all_pred_boxes)
    true_boxes = torch.stack(all_true_boxes)
    
    if is_test:
        return true_boxes, pred_boxes, all_paths
    return true_boxes, pred_boxes

def train_evaluate_test_detection_model(model, device, train_loader, val_loader, test_loader,
                                      num_epochs=5, lr=1e-3, results_dir="./results", models_dir="./models",
                                      early_stopping_threshold=95.0):
    """Handles complete training cycle for detection models.
    
    Features:
    - Adam optimiser
    - IoU-based early stopping
    - Best model checkpointing
    - Training progress plots
    - Test set evaluation with bounding box metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    model_name = model.__class__.__name__
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_iou = 0
    history = {
        'train_loss': [],
        'val_iou': []
    }

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_one_epoch_detection(model, device, train_loader, optimizer)
        val_true_boxes, val_pred_boxes = validate_one_epoch_detection(model, device, val_loader)
        
        # Calculate mean IoU for validation set
        val_ious = ops.box_iou(val_pred_boxes, val_true_boxes).diagonal()
        val_iou = val_ious.mean().item() * 100

        end_time = time.time()
        epoch_time = end_time - start_time

        history['train_loss'].append(train_loss)
        history['val_iou'].append(val_iou)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val IoU: {val_iou:.2f}% "
              f"Time: {epoch_time:.2f}s")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), os.path.join(models_dir, f"{model_name}_best.pth"))

        if val_iou >= early_stopping_threshold:
            print(f"\nReached IoU threshold of {early_stopping_threshold}% at epoch {epoch+1}")
            break

    plot_training_progress(
        history,
        os.path.join(results_dir, f"{model_name}_training_progress.png")
    )

    # Evaluate on test set using best model
    model.load_state_dict(torch.load(os.path.join(models_dir, f"{model_name}_best.pth")))
    test_true_boxes, test_pred_boxes, test_paths = validate_one_epoch_detection(
        model, device, test_loader, is_test=True
    )

    metrics = plot_detection_metrics(
        test_pred_boxes,
        test_true_boxes,
        os.path.join(results_dir, f"{model_name}_analysis.png")
    )

    save_detection_predictions(
        test_paths,
        test_pred_boxes,
        test_true_boxes,
        metrics,
        os.path.join(results_dir, f"{model_name}_test_predictions.csv")
    )

    return history, (val_true_boxes, val_pred_boxes), (test_true_boxes, test_pred_boxes)
