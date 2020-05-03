"""
Neural Network models for training and testing implemented in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

# Standard 4 layer CNN implementation
class CNN(nn.Module):

    # 2 Conv layers, 2 Fc layers

    def __init__(self):
        super(CNN, self).__init__()

        self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5, stride = 1, padding = 2, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = 2, bias=True)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 10, bias=True)
       

    def forward(self, x):

        x = self.norm(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



class FcNN(nn.Module):

    def __init__(self):
        super(FcNN, self).__init__()

        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 10)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Fc5NN(nn.Module):

    def __init__(self):
        super(Fc5NN, self).__init__()
        # 1 input image channel, 200 output channels, 5x5 square convolution
        # kernel
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 1000)
        self.fc5 = nn.Linear(1000, 1000)
        self.fc6 = nn.Linear(1000, 10)

    # If you want to see sparsity of neuron outputs for each layer, set show_sparsity = True
    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        return x
