#!/usr/bin/env python3

import os, sys 

sys.path.append("/Users/KerimErekmen/Desktop/chesstwitter/lib/")
sys.path.append("/Users/KerimErekmen/Desktop/chesstwitter/lib/python3.9/site-packages/sklearn/")
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as function_
from torch import optim
import torch.nn as nn
from sklearn.model_selection import train_test_split


class CustomImageDataset(Dataset):   
    def __init__(self, file_dir):
        self.con = True # for every object disjunct
        if str(file_dir).endswith(".npz"):
            self.con = False
            file_data = np.load(file_dir)
            self.x = file_data['arr_0']
            self.y = file_data['arr_1']
        else:
            self.x = file_dir # is transformed
            self.y = file_dir

        print("loaded file")

    def __len__(self):
        """number of samples in our dataset

        Returns:
            int: number of samples
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """loads and returns a sample from the dataset at the given index idx

        Args:
            idx (int): index of sample

        Returns:
            sample, label = x, y
        """
        if self.con:
            return (self.x[idx][0], self.y[idx][1])
        else:
            return (self.x[idx], self.y[idx])
        
        return None

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
        x = function_.relu(self.a1(x))
        x = function_.relu(self.a2(x))
        x = function_.relu(self.a3(x))

        x = function_.relu(self.b1(x))
        x = function_.relu(self.b2(x))
        x = function_.relu(self.b3(x))

        x = function_.relu(self.d1(x))
        x = function_.relu(self.d2(x))
        x = function_.relu(self.d3(x))

        x = x.view(-1, 128)
        x = function_.tanh(self.last(x))

        return x

    def train(self, model, optim, loader, criterion, epochs):
        batches = len(loader)
        for epoch in range(1, epochs+1):
            running_loss = 0
            correct_acc = 0 # TP+FP/TP+FP+TN+FN
            total_loss = 0
            print(f"Epoch {epoch} \n")
            for i, (data, labels) in enumerate(loader):
                data = data.to(dtype=torch.float)
                labels = labels.to(dtype=torch.float)

                optim.zero_grad()
                output = model(data)
                loss = criterion(output, labels.unsqueeze(-1))
                loss.backward()
                optim.step()

                running_loss += loss.item()
                total_loss += 1

                if i % 1000 == 0:
                    print(f"Batch [{i}/{batches}], Loss [{loss.item():.4f}], outputmin {output.min()}, outputmax {output.max()}")
            print(f"Total Loss of {running_loss/total_loss:.4f} for epoch {epoch}")




if __name__ == "__main__":
    chess_dataset = CustomImageDataset("chess_nn_data.npz")
    dataset = np.array(list(zip(list(chess_dataset.x), list(chess_dataset.y))))
    #print(dataset.shape)
    
    train_set, test_set = train_test_split(dataset, test_size=0.25)
    train_set = CustomImageDataset(train_set)
    test_set = CustomImageDataset(test_set)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)
    
    model = NN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train(model=model, optim=optimizer, loader=train_loader, criterion=criterion, epochs=10)
    """
    for data, target in train_loader:
        print(data.shape, target.shape)
    """

    