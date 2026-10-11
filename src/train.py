"""Train a simple CNN on 2 image classes (bibimbap vs fried_rice).
"""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing_data import create_dataloaders
from model import SimpleCNN, print_model_summary
from utils import MODELS_DIR, PLOTS_DIR, get_device, plot_training_history, save_json, set_seed


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch. Returns (loss, accuracy)."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Weight the batch loss by batch size, because the last batch can be smaller
        running_loss += loss.item() * images.size(0)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    """Evaluate the model for one epoch without training. Returns (loss, accuracy)."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train a simple CNN on 2 image classes.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Adam learning rate.")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print("Using device:", device)

    # torch.save does not create folders, so make sure they exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, class_names = create_dataloaders(batch_size=args.batch_size)

    model = SimpleCNN(num_classes=2).to(device)
    print_model_summary(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    model_save_path = MODELS_DIR / "best_cnn_model.pth"

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save the model only when validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model to {model_save_path}")

    save_json(history, PLOTS_DIR / "training_history.json")
    save_json(class_names, MODELS_DIR / "class_names.json")
    plot_training_history(history, PLOTS_DIR / "training_curves.png")

    print("\nTraining complete.")
    print("Best validation accuracy:", round(best_val_acc, 4))
    print("Saved model to:", model_save_path)
    print("Saved class names to:", MODELS_DIR / "class_names.json")
    print("Saved plots to:", PLOTS_DIR / "training_curves.png")


if __name__ == "__main__":
    main()