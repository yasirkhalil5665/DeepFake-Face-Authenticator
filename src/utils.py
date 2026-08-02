"""Utility functions: saving trained models, plotting/saving loss+accuracy curves
as PNGs, and saving a sample-prediction grid as a PNG."""
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torch import nn


def save_model(model: nn.Module, target_dir: str, model_name: str) -> Path:
    """Saves a model's state_dict to `target_dir/model_name`."""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
    return model_save_path


def plot_curves(results: Dict[str, List[float]], save_path: str = None):
    """Plots train/val loss and accuracy curves. Saves as PNG if `save_path` is given."""
    epochs = range(len(results["train_loss"]))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(epochs, results["train_loss"], label="train_loss")
    ax[0].plot(epochs, results["val_loss"], label="val_loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(epochs, results["train_acc"], label="train_acc")
    ax[1].plot(epochs, results["val_acc"], label="val_acc")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Accuracy")
    ax[1].legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"[INFO] Saved curves plot to: {save_path}")

    plt.show()
    return fig


def save_prediction_grid(model: nn.Module, dataset, class_names: List[str], device: str,
                          save_path: str, num_images: int = 6, seed: int = 42):
    """Runs the model on random samples from `dataset` and saves a labeled
    prediction grid as a PNG (green title = correct, red = incorrect)."""
    random.seed(seed)
    indices = random.sample(range(len(dataset)), k=min(num_images, len(dataset)))

    model.eval()
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        img, label = dataset[idx]
        with torch.inference_mode():
            pred = model(img.unsqueeze(0).to(device))
            pred_label = torch.argmax(torch.softmax(pred, dim=1), dim=1).item()

        ax.imshow(img.permute(1, 2, 0).cpu())
        color = "green" if pred_label == label else "red"
        ax.set_title(f"Pred: {class_names[pred_label]}\nTrue: {class_names[label]}", color=color)
        ax.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    print(f"[INFO] Saved prediction grid to: {save_path}")
    plt.show()
    return fig
