"""
model.py

Defines the Convolutional Neural Network architecture used to classify
anime character face images.
"""

import torch.nn as nn
import torch.nn.functional as F


class AnimeCNN(nn.Module):
    """
    A simple CNN for binary (or small multi-class) image classification.

    Architecture:
        Conv(3->32, 3x3) -> ReLU -> MaxPool(2x2)
        Conv(32->64, 3x3) -> ReLU -> MaxPool(2x2)
        Flatten
        Linear(64*16*16 -> 128) -> ReLU
        Linear(128 -> num_classes)

    Assumes input images are 64x64. If you change the input resolution,
    update fc1's input size accordingly (64 * (H/4) * (W/4)).
    """

    def __init__(self, num_classes=2, activation="relu"):
        super(AnimeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
