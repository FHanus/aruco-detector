import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from utils.model_analysis import plot_classification_metrics, plot_training_progress, save_test_predictions

def train_one_epoch(model, device, train_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for _, (inputs, targets, _) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        # Accuracy
        _, predicted = torch.max(outputs, 1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += inputs.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = 100. * total_correct / total_samples

    return epoch_loss, epoch_acc


def validate_one_epoch(model, device, val_loader, loss_fn, is_test=False):
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

            # Accuracy
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
                       early_stopping_threshold=99.):
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    model_name = model.__class__.__name__
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, device, train_loader, loss_fn, optimizer)
        val_loss, val_acc, val_labels, val_preds = validate_one_epoch(model, device, val_loader, loss_fn)

        end_time = time.time()
        epoch_time = end_time - start_time

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}% "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}% "
              f"Time: {epoch_time:.2f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(models_dir, f"{model_name}_best.pth"))

         # Early stopping check
        if val_acc >= early_stopping_threshold:
            print(f"\nReached validation accuracy threshold of {early_stopping_threshold:.2%} at epoch {epoch+1} ({val_acc:.2%}).")
            print("Early stopping triggered.")
            break

    # Plot training progress
    plot_training_progress(
        history, 
        os.path.join(results_dir, f"{model_name}_training_progress.png")
    )

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(models_dir, f"{model_name}_best.pth")))

    test_loss, test_acc, test_labels, test_preds, test_paths = validate_one_epoch(model, device, test_loader, loss_fn, is_test=True)

    print(f"\nTest Set Performance:")
    print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}%")

    # Save analysis plots and get metrics
    metrics = plot_classification_metrics(
        test_labels, 
        test_preds,
        os.path.join(results_dir, f"{model_name}_analysis.png")
    )
    
    # Save test predictions to CSV with metrics
    save_test_predictions(
        test_paths,
        test_labels,
        test_preds,
        metrics,
        os.path.join(results_dir, f"{model_name}_test_predictions.csv")
    )

    return history, (val_labels, val_preds), (test_labels, test_preds)