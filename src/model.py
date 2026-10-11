import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            # Input shape: [batch_size, 3, 128, 128]
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # out = (image_size - kernel_size + 2 * padding) // stride + 1
            # Output shape after pool: [batch_size, 16, 64, 64]

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output shape after pool: [batch_size, 32, 32, 32]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 32 * 32 * 32 = 32768 features for each image
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.classifier(x)
        return x


def print_model_summary(model: nn.Module):
    """Print a simple summary that is easy for beginners to read."""
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print("\nModel structure:")
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")