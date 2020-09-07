"""
Neural Network models for training and testing implemented in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .tools import Normalize


class LeNet(nn.Module):

    # 2 Conv layers, 2 Fc layers

    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()

        self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

    def forward(self, x):

        out = self.norm(x)
        out = F.max_pool2d(F.relu(self.conv1(out)), (2, 2))
        out = F.max_pool2d(F.relu(self.conv2(out)), (2, 2))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


class LeNet2d(nn.Module):

    # 2 Conv layers, 2 Fc layers

    def __init__(self, num_classes=10):
        super(LeNet2d, self).__init__()

        self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.fc1 = nn.Linear(7 * 7 * 64, 2, bias=True)
        self.fc2 = nn.Linear(2, num_classes, bias=True)

    def forward(self, x):

        out = self.norm(x)
        out = F.max_pool2d(F.relu(self.conv1(out)), (2, 2))
        out = F.max_pool2d(F.relu(self.conv2(out)), (2, 2))
        out = out.view(out.size(0), -1)
        embedding = self.fc1(out)
        out = self.fc2(embedding)

        return out, embedding
