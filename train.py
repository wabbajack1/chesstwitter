#!/usr/bin/env python3

import os, sys 

sys.path.append("lib/")
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as function
from torch import optim
import torch.nn as nn


class CustomImageDataset(Dataset):
    def __init__(self, file_dir):
        self.file_data = np.load(file_dir)
        self.file_dir = file_dir
        print("loaded file")

    def __len__(self):
        """number of samples in our dataset

        Returns:
            int: number of samples
        """
        return self.file_data["arr_0"].shape[0]

    def __getitem__(self, idx):
        """loads and returns a sample from the dataset at the given index idx

        Args:
            idx (int): index of sample

        Returns:
            sample, label
        """
        position = self.file_data["arr_0"][idx]
        label = self.file_data["arr_1"][idx]

        return (position, label)

class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()
        self.a1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        self.b1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)

        self.d1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.d2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.d3 = nn.Conv2d(128, 128, kernel_size=1)

        self.last = nn.Linear(128, 1)

    def forward(self, x):
        x = function.relu(self.a1(x))
        x = function.relu(self.a2(x))
        x = function.relu(self.a3(x))

        x = function.relu(self.b1(x))
        x = function.relu(self.b2(x))
        x = function.relu(self.b3(x))

        x = function.relu(self.d1(x))
        x = function.relu(self.d2(x))
        x = function.relu(self.d3(x))

        x = torch.flatten(x)
        x = function.sigmoid(self.last(x))

        return x



if __name__ == "__main__":
    dataset = CustomImageDataset("chess_nn_data.npz")

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    model = NN()
    optimizer = optim.Adam(model.parameters())
    loss = nn.MSELoss()

    print("here")
    for i in range(train_loader):
        print(i)

    """
    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    """