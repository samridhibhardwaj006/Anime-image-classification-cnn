"""
dataset.py

Handles downloading/loading the anime face dataset from a zip file
and defines the custom PyTorch Dataset used to feed images into the CNN.
"""

import io
import zipfile

import numpy as np
import requests
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_images_from_zip(zip_source, classes):
    """
    Load images belonging to `classes` out of a zip file.

    Args:
        zip_source: a file-like object (e.g. io.BytesIO) or a path to a .zip file.
        classes: list of class names. Each class name is assumed to be a prefix
                 of the image filenames inside the zip (e.g. 'anastasia_01.jpg').

    Returns:
        dict mapping class_name -> list of images as numpy arrays (RGB).
    """
    images = {class_name: [] for class_name in classes}

    with zipfile.ZipFile(zip_source, "r") as zip_ref:
        for file_name in zip_ref.namelist():
            if not file_name.endswith(".jpg"):
                continue
            for class_name in classes:
                if file_name.startswith(class_name):
                    with zip_ref.open(file_name) as file:
                        img = Image.open(file).convert("RGB")
                        images[class_name].append(np.array(img))
                    break

    return images


def download_and_load_images(zip_file_url, classes):
    """
    Download a zip file from a URL and load images for the given classes.

    Args:
        zip_file_url: URL pointing to a .zip file of images.
        classes: list of class names (used as filename prefixes).

    Returns:
        dict mapping class_name -> list of images as numpy arrays (RGB).
    """
    response = requests.get(zip_file_url)
    response.raise_for_status()
    zip_file_bytes = io.BytesIO(response.content)
    return load_images_from_zip(zip_file_bytes, classes)


class AnimeDataset(Dataset):
    """
    Custom PyTorch Dataset for anime character face images.

    Args:
        images: dict mapping class_name -> list of numpy-array images.
        transform: torchvision transform(s) to apply to each image.
        classes: ordered list of class names. The index of a class name in
                 this list becomes its integer label.
    """

    def __init__(self, images, transform=None, classes=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.classes = classes

        for label, class_name in enumerate(self.classes):
            for img in images[class_name]:
                self.images.append(img)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_default_transform(image_size=64):
    """
    Returns the default image transform pipeline used throughout the project:
    resize -> tensor -> normalize to [-1, 1] per channel.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
