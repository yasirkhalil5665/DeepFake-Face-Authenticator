"""Trains a real-vs-fake face classifier end-to-end and saves the model +
result plots (loss/accuracy curves PNG and a sample-predictions PNG).

Example usage:
    python src/train.py --model efficientnet --epochs 5 --batch-size 32
    python src/train.py --model cnn --epochs 10 --image-size 128
    python src/train.py --model resnet18 --epochs 5
"""
import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms

import data_setup
import engine
import model_builder
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Train a real-vs-fake face classifier.")
    parser.add_argument("--data-dir", type=str, default="data/deepfakefusion_v2_sixsource_dataset",
                         help="Root folder containing train/ val/ test/ subfolders (ImageFolder format).")
    parser.add_argument("--model", type=str, default="efficientnet",
                         choices=["cnn", "resnet18", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    data_dir = Path(args.data_dir)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])

    train_dataloader, val_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=data_dir / "train",
        val_dir=data_dir / "val",
        test_dir=data_dir / "test",
        transform=transform,
        batch_size=args.batch_size,
    )
    print(f"[INFO] Classes: {class_names}")

    if args.model == "cnn":
        model = model_builder.TinyVGG(
            output_shape=len(class_names), image_size=args.image_size
        ).to(device)
    elif args.model == "resnet18":
        model = model_builder.create_resnet18(output_shape=len(class_names), device=device)
    else:
        model = model_builder.create_efficientnet_b0(output_shape=len(class_names), device=device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device,
    )

    engine.evaluate_model(model, test_dataloader, loss_fn, device)

    utils.plot_curves(results, save_path=f"{args.results_dir}/{args.model}_curves.png")
    utils.save_prediction_grid(
        model, test_dataloader.dataset, class_names, device,
        save_path=f"{args.results_dir}/{args.model}_predictions.png",
    )
    utils.save_model(model, target_dir=args.models_dir, model_name=f"{args.model}_model.pth")

if __name__ == "__main__":
    main()
