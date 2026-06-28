"""
main.py

End-to-end entry point: downloads data, builds the dataset, trains the
AnimeCNN model, plots the loss curves, and visualizes predictions.

Usage:
    python src/main.py
    python src/main.py --epochs 10 --activation leaky_relu
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import AnimeDataset, download_and_load_images, get_default_transform
from evaluate import plot_images, visualize_predictions
from model import AnimeCNN
from train import make_dataloaders, plot_losses, set_seed, train_model

DEFAULT_ZIP_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "xZQHOyN8ONT92kH-ASb4Pw/data.zip"
)
DEFAULT_CLASSES = ["anastasia", "takao"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train an anime face classifier CNN.")
    parser.add_argument("--zip-url", type=str, default=DEFAULT_ZIP_URL,
                         help="URL of the zip file containing class-labeled images.")
    parser.add_argument("--classes", type=str, nargs="+", default=DEFAULT_CLASSES,
                         help="Class names; also used as filename prefixes in the zip.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--activation", type=str, default="relu",
                         choices=["relu", "leaky_relu"], help="Activation function to use.")
    parser.add_argument("--image-size", type=int, default=64, help="Resize images to this size.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation set fraction.")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-visualize", action="store_true",
                         help="Skip the raw-image and prediction visualizations.")
    parser.add_argument("--output-dir", type=str, default="outputs",
                         help="Directory to save the trained model and loss plot.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    set_seed(args.seed)

    print("Downloading and loading images...")
    images = download_and_load_images(args.zip_url, args.classes)
    for class_name in args.classes:
        print(f"Number of images of {class_name}: {len(images[class_name])}")

    if not args.no_visualize:
        for class_name in args.classes:
            plot_images(images[class_name], f"{class_name} Images")

    transform = get_default_transform(image_size=args.image_size)
    dataset = AnimeDataset(images, transform=transform, classes=args.classes)

    train_loader, val_loader, train_idx, val_idx = make_dataloaders(
        dataset,
        val_split=args.val_split,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        seed=args.seed,
    )
    print("Train size:", len(train_idx))
    print("Validation size:", len(val_idx))

    model = AnimeCNN(num_classes=len(args.classes), activation=args.activation)
    print(model)

    train_losses, val_losses = train_model(
        model, train_loader, val_loader, num_epochs=args.epochs, lr=args.lr, device=device
    )

    loss_plot_path = os.path.join(args.output_dir, "loss_curve.png")
    plot_losses(train_losses, val_losses, save_path=loss_plot_path)

    model_path = os.path.join(args.output_dir, "anime_cnn.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")

    if not args.no_visualize:
        visualize_predictions(model, val_loader, device=device)


if __name__ == "__main__":
    main()
