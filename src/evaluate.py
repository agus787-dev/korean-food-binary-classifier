import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from preprocessing_data import create_dataloaders
from model import SimpleCNN
from utils import MODELS_DIR, PLOTS_DIR, get_device, load_json, plot_confusion_matrix


def main():
    device = get_device()
    print("Using device:", device)

    _, _, test_loader, _ = create_dataloaders(batch_size=16)

    class_names = load_json(MODELS_DIR / "class_names.json")
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_cnn_model.pth", map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_predictions = []
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    test_loss = running_loss / total
    test_accuracy = accuracy_score(all_labels, all_predictions)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    confusion_matrix_path = PLOTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(
        y_true=all_labels,
        y_pred=all_predictions,
        class_names=class_names,
        save_path=confusion_matrix_path,
    )
    print("Saved confusion matrix to:", confusion_matrix_path)


if __name__ == "__main__":
    main()