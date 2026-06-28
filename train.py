"""
train.py

Training loop and loss-curve plotting utilities for the AnimeCNN model.
Can be run as a standalone script (see __main__ block) or imported
and used programmatically.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import (
    AnimeDataset,
    download_and_load_images,
    get_default_transform,
)
from model import AnimeCNN


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataloaders(dataset, val_split=0.2, train_batch_size=8, val_batch_size=20, seed=42):
    """
    Split a dataset into train/validation sets and wrap each in a DataLoader.

    Returns:
        train_loader, val_loader, train_indices, val_indices
    """
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, random_state=seed
    )

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=train_batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=val_batch_size, sampler=val_sampler)

    return train_loader, val_loader, train_indices, val_indices


def train_model(model, train_loader, val_loader, num_epochs=5, lr=0.001, device="cpu"):
    """
    Train `model` on `train_loader` and evaluate on `val_loader` each epoch.

    Returns:
        train_losses, val_losses (lists of per-epoch average loss)
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("Finished Training")
    return train_losses, val_losses


def plot_losses(train_losses, val_losses, save_path=None):
    """Plot training vs validation loss curves. Optionally save to disk."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training and Validation Loss")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved loss plot to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Default config: reproduces the original notebook's Anastasia vs. Takao task.
    ZIP_URL = (
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "xZQHOyN8ONT92kH-ASb4Pw/data.zip"
    )
    CLASSES = ["anastasia", "takao"]
    NUM_EPOCHS = 5
    SEED = 42

    set_seed(SEED)

    print("Downloading and loading images...")
    images = download_and_load_images(ZIP_URL, CLASSES)
    for class_name in CLASSES:
        print(f"Number of images of {class_name}: {len(images[class_name])}")

    transform = get_default_transform(image_size=64)
    dataset = AnimeDataset(images, transform=transform, classes=CLASSES)

    train_loader, val_loader, train_idx, val_idx = make_dataloaders(dataset, seed=SEED)
    print("Train size:", len(train_idx))
    print("Validation size:", len(val_idx))

    model = AnimeCNN(num_classes=len(CLASSES))
    print(model)

    train_losses, val_losses = train_model(
        model, train_loader, val_loader, num_epochs=NUM_EPOCHS
    )

    plot_losses(train_losses, val_losses, save_path="outputs/loss_curve.png")

    torch.save(model.state_dict(), "outputs/anime_cnn.pth")
    print("Saved trained model to outputs/anime_cnn.pth")
