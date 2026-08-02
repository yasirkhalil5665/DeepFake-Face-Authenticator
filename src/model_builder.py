"""PyTorch model definitions: a small custom CNN, plus transfer-learning builders."""
from torch import nn
from torchvision import models


class TinyVGG(nn.Module):
    """A small from-scratch CNN (2 conv blocks) for image classification."""

    def __init__(self, input_shape: int = 3, hidden_units: int = 12,
                 output_shape: int = 2, image_size: int = 128):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        flattened_size = hidden_units * (image_size // 4) * (image_size // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, output_shape),
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


def create_efficientnet_b0(output_shape: int = 2, device: str = "cpu") -> nn.Module:
    """Loads a pretrained EfficientNet-B0 and replaces the classifier head."""
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_shape)
    return model.to(device)


def create_resnet18(output_shape: int = 2, device: str = "cpu") -> nn.Module:
    """Loads a pretrained ResNet18 and replaces the final fully-connected layer."""
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, output_shape)
    return model.to(device)
