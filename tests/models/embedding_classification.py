"""
Attention Layers
"""

import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiModeEmbeddingMaxpooling(nn.Module):
    """Data normalizing class as torch.nn.Module

    Attributes:
        num_modes (int): Number of parallel linear layers.
        input_size (int): Input size of parallel linear layers.
        num_classes (int): Number of classes (outputs).
        parallel_layers (nn.ModuleList): Parallel linear layers.
        maxpooling (nn.MaxPool1d): Max-pooling layer.

    """

    def __init__(self, num_modes, input_size, num_classes=10):
        """

        Args:
            num_modes (int): Number of parallel linear layers.
            input_size (int): Input size of parallel linear layers.
            num_classes (int): Number of classes (outputs).

        """
        super(MultiModeEmbeddingMaxpooling, self).__init__()

        self.input_size = input_size
        self.num_modes = num_modes
        self.num_classes = num_classes

        parallel_layers = self._build_parallel_layers()
        self.parallel_layers = nn.ModuleList(parallel_layers)

        self.maxpooling = nn.MaxPool1d(kernel_size=num_modes, stride=num_modes, return_indices=True)

    def forward(self, x):
        """
        Args:
            x (tensor batch): Input tensor.
        Returns:
            out (tensor batch): Output logits.
        """
        out = [None] * self.num_classes
        for i, layer in enumerate(self.parallel_layers):
            out[i] = layer(x).unsqueeze(dim=1)
        out = torch.cat(tuple(out), dim=1)
        out, indices = self.maxpooling(out)
        out = out.squeeze()

        return out  # , indices

    def _build_parallel_layers(self):
        """build parallel layers"""

        parallel_layers = [None] * self.num_classes

        for i in range(self.num_classes):
            parallel_layers[i] = nn.Linear(
                self.input_size, self.num_modes, bias=True)
            torch.nn.init.xavier_normal_(parallel_layers[i].weight)

        return parallel_layers


class MultiModeEmbeddingClassification(nn.Module):
    """Data normalizing class as torch.nn.Module

    Attributes:
        num_modes (int): Number of parallel linear layers.
        input_size (int): Input size of parallel linear layers.
        num_classes (int): Number of classes (outputs).
        parallel_layers (nn.ModuleList): Parallel linear layers.
        partial_sum (nn.Module): Per class summer.

    """

    def __init__(self, num_modes, input_size, num_classes=10):
        """

        Args:
            num_modes (int): Number of parallel linear layers.
            input_size (int): Input size of parallel linear layers.
            num_classes (int): Number of classes (outputs).

        """
        super(MultiModeEmbeddingClassification, self).__init__()
        self.input_size = input_size
        self.num_modes = num_modes
        self.num_classes = num_classes

        parallel_layers = self._build_parallel_layers()
        self.parallel_layers = nn.ModuleList(parallel_layers)

        self.partial_sum = PartialSum(num_modes, self.num_classes)

    def forward(self, x):
        """
        Args:
            x (tensor batch): Input tensor.
        Returns:
            out (tensor batch): Output logits.
        """
        out = [None] * self.num_classes
        for i, layer in enumerate(self.parallel_layers):
            out[i] = layer(x)
        out = torch.cat(tuple(out), -1)
        out = self.partial_sum(out)

        return out

    def _build_parallel_layers(self):
        """build parallel layers"""

        parallel_layers = [None] * self.num_classes

        for i in range(self.num_classes):
            parallel_layers[i] = nn.Linear(
                self.input_size, self.num_modes, bias=True)
            torch.nn.init.xavier_normal_(parallel_layers[i].weight)

        return parallel_layers


class PartialSum(nn.Module):

    def __init__(self, partition_length, num_partitions, device=torch.device("cuda")):
        super(PartialSum, self).__init__()
        self.partition_length = partition_length
        self.num_partitions = num_partitions

        weights = torch.tensor(np.ones((1, partition_length))).to(device)

        self.summers = [None] * num_partitions
        for i in range(num_partitions):
            self.summers[i] = nn.Linear(partition_length, 1, bias=False).to(device)
            with torch.no_grad():
                self.summers[i].weight.copy_(weights)
                for param in self.summers[i].parameters():
                    param.requires_grad = False

    def forward(self, x):

        out = [None] * self.num_partitions
        for i in range(self.num_partitions):
            out[i] = self.summers[i](x[:, i*self.partition_length:(i+1)*self.partition_length])
        out = torch.cat(tuple(out), -1)

        return out


class Combined(nn.Module):

    def __init__(self, module_inner, module_outer):
        super(Combined, self).__init__()
        self.module_inner = module_inner
        for param in self.module_inner.parameters():
            param.requires_grad = False
        self.module_outer = module_outer

    def forward(self, input):
        return self.module_outer(self.module_inner(input))
