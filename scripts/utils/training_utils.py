import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from utils.model_analysis import plot_classification_metrics, plot_training_progress

def train_one_epoch(model, device, train_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for _, (inputs, targets) in enumerate(train_loader):
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


def validate_one_epoch(model, device, val_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(val_loader):
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

    epoch_loss = total_loss / total_samples
    epoch_acc = 100. * total_correct / total_samples

    return epoch_loss, epoch_acc, all_labels, all_preds

def train_evaluate_test_model(model, device, train_loader, val_loader, test_loader,
                       num_epochs=5, lr=1e-3, results_dir="./results", models_dir="./models"):
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

    # Save final model
    torch.save(model.state_dict(), os.path.join(models_dir, f"{model_name}_final.pth"))

    # Plot training progress
    plot_training_progress(
        history, 
        os.path.join(results_dir, f"{model_name}_training_progress.png")
    )

    # Load model and test
    model.load_state_dict(torch.load(os.path.join(models_dir, f"{model_name}_best.pth")))

    test_loss, test_acc, test_labels, test_preds = validate_one_epoch(model, device, test_loader, loss_fn)

    print(f"\nTest Set Performance:")
    print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}%")

    # Save analysis plots
    plot_classification_metrics(
        test_labels, 
        test_preds,
        os.path.join(results_dir, f"{model_name}_analysis.png")
    )

    return history, (val_labels, val_preds), (test_labels, test_preds)