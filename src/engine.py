"""Training and evaluation loops for a PyTorch image classifier."""
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer, device: str) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def test_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module,
               device: str) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0.0, 0.0

    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_class = torch.argmax(y_pred, dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(y_pred)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc


def train(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer, loss_fn: nn.Module, epochs: int,
          device: str, start_epoch: int = 0, results: Dict[str, List[float]] = None,
          checkpoint_dir: str = None, checkpoint_name: str = "checkpoint.pth") -> Dict[str, List[float]]:
    import utils 

    if results is None:
        results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(start_epoch, epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_acc = test_step(model, val_dataloader, loss_fn, device)

        print(
            f"Epoch {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc * 100:.2f}% | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc * 100:.2f}%"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        if checkpoint_dir:
            utils.save_checkpoint(model, optimizer, epoch, results, checkpoint_dir, checkpoint_name)

    return results


def evaluate_model(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module,
                    device: str) -> Tuple[float, float]:
    """Evaluates the model on a held-out dataloader (typically the test set)."""
    test_loss, test_acc = test_step(model, dataloader, loss_fn, device)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc * 100:.2f}%")
    return test_loss, test_acc
