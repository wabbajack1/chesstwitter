#!/usr/bin/env python3

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch import optim


class CustomImageDataset(Dataset):
    def __init__(self, file_dir):
        self.file_data = np.load(file_dir)
        self.file_dir = file_dir

    def __len__(self):
        """number of samples in our dataset

        Returns:
            int: number of samples
        """
        return len(self.file_data["arr_0"].size)

    def __getitem__(self, idx):
        """loads and returns a sample from the dataset at the given index idx

        Args:
            idx (int): index of sample

        Returns:
            sample, label
        """
        position = self.file_data["arr_0"][idx]
        label = self.file_data["arr_1"][idx]

        return position, label

class NN(torch.nn.Module):

    def __init__(self):
        super(NN, self).__init__()
        self.a1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        self.b1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)

        self.d1 = nn.Conv2d(128, 128, kernel_size=1)
        self.d2 = nn.Conv2d(128, 128, kernel_size=1)
        self.d3 = nn.Conv2d(128, 128, kernel_size=1)

        self.last = nn.Linear(128, 1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    dataset = CustomImageDataset("chess_nn_data.npz")