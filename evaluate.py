"""
evaluate.py

Utilities for visualizing model predictions on the validation set:
- showing a grid of images with their dataset class indices (Anastasia/Takao)
- showing predicted vs. actual labels side-by-side
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def imshow(img, ax):
    """Unnormalize a tensor image (assumed normalized to [-1, 1]) and display it."""
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.axis("off")


def plot_images(images, title):
    """
    Display a grid of raw (numpy array) images, e.g. straight from the
    loaded dataset before any tensor transforms.
    """
    fig, axes = plt.subplots(5, 10, figsize=(10, 5))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, val_loader, device="cpu", num_cols=10):
    """
    Run the model on one batch from val_loader and plot each image next to
    its actual and predicted class label.
    """
    model.eval()
    model.to(device)

    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    images_device = images.to(device)

    with torch.no_grad():
        outputs = model(images_device)
        _, predicted = torch.max(outputs, 1)

    num_images = len(images)
    num_rows = max(1, -(-num_images // num_cols))  # ceil division

    fig, axs = plt.subplots(num_rows, num_cols * 2, figsize=(20, num_rows * 2))
    if num_rows == 1:
        axs = np.expand_dims(axs, axis=0)

    for idx in range(num_images):
        row = idx // num_cols
        col = (idx % num_cols) * 2

        imshow(images[idx].cpu(), axs[row, col])
        axs[row, col + 1].text(
            0.5,
            0.5,
            f"Actual: {labels[idx].item()}\nPredicted: {predicted[idx].item()}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )
        axs[row, col + 1].axis("off")

    # Hide any unused axes
    for idx in range(num_images, num_rows * num_cols):
        row = idx // num_cols
        col = (idx % num_cols) * 2
        axs[row, col].axis("off")
        axs[row, col + 1].axis("off")

    plt.tight_layout()
    plt.show()
