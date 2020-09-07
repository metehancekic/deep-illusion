"""
Attention Layers
"""

import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F


class Normalize(nn.Module):
    """Data normalizing class as torch.nn.Module

    Attributes:
        mean (float): Mean value of the training dataset.
        std (float): Standard deviation value of the training dataset.

    """

    def __init__(self, mean, std):
        """

        Args:
            mean (float): Mean value of the training dataset.
            std (float): Standard deviation value of the training dataset.

        """
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        """
        Args:
            x (tensor batch): Input tensor.
        Returns:
            Normalized data.
        """
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class ScaledDotProductAttention(nn.Module):
    """Scaled dot product attention from "Attention is all you need" paper

    Attributes:
        scaling_factor (float): Scaling factor for dot product of query and key embedding.
        attention_dropout (:obj:`float`, optional): dropping rate for dropout.

    """

    def __init__(self, scaling_factor, attention_dropout=0.1):
        """

        Args:
            scaling_factor (float): Scaling factor for dot product of query and key embedding.
            attention_dropout (:obj:`float`, optional): dropping rate for dropout.

        """
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query, key, value):
        """

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor

        Returns:
            output: Attended output
            attention: Attention tensor

        """

        attention = torch.matmul(query / self.scaling_factor, key.transpose(1, 2))
        attention = self.dropout(F.softmax(attention, dim=-1))
        output = torch.matmul(attention, value)

        return output, attention


class VanillaMLPAttention(nn.Module):
    """Vanilla attention class with MLP

    Attributes:
            feature_size (int): Feature size that is gonna fed as an input.
            encoder (nn.Linear): First layer of MLP.
            decoder (nn.Linear): Second layer of MLP.

    """

    def __init__(self, feature_size, encoding_size=16, attention_dropout=0.1):
        """

        Args:
                feature_size (int): Feature size that is gonna fed as an input.
                encoding_size (int, default=16): Number of neurons in the hidden layer.
                attention_dropout: (float, default=0.1): dropping rate for dropout.

        """
        super().__init__()
        self.feature_size = feature_size
        self.encoder = nn.Linear(feature_size, encoding_size, bias=True)
        self.decoder = nn.Linear(encoding_size, feature_size, bias=True)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x):
        """

        Args:
                x (tensor): Input tensor.

        Returns:
                out: Attended output.
                attention: Attention tensor.

        """
        out = F.relu(self.encoder(x))
        out = F.relu(self.decoder(out))
        attention = F.softmax(out, dim=-1)
        out = attention * out

        return out, attention


class VanillaConvolutionalAttention(nn.Module):
    """Vanilla attention class with Convolutional

    Attributes:
        number_channels (int): Number of channels in the input.
        attention_cnn (torch.nn.Conv2d): Attention learning cnn.

    """

    def __init__(self, number_channels, attention_dropout=0.1):
        """

        Args:
            number_channels (int): Number of channels in the input.
            attention_dropout: (float, default=0.1): dropping rate for dropout.

        """
        super().__init__()
        self.number_channels = number_channels
        self.attention_cnn = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=3,
                      stride=1, padding=1, groups=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(number_channels, number_channels, kernel_size=3,
                      stride=1, padding=1, groups=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(number_channels, number_channels, kernel_size=3,
                      stride=1, padding=1, groups=1, bias=False)
            )

    def forward(self, x):
        """

        Args:
            x (tensor): Input tensor.

        Returns:
            out: Attended output.
            attention: Attention tensor.

        """
        scores = self.attention_cnn(x)
        attention = nn.Softmax(2)(scores.view(*scores.size()[:2], -1)).view_as(scores)
        out = attention * x

        return out, attention


class FeatureAttention(nn.Module):
    ''' Scaled Dot-Product Attention
            "All you need is attention"
    '''

    def __init__(self, feature_size, num_filters=64, dropout=0.1):
        super().__init__()

        self.query_linear = nn.Linear(feature_size, feature_size, bias=False)
        self.key_linear = nn.Linear(feature_size, feature_size, bias=False)
        self.value_conv = nn.Conv2d(num_filters, num_filters, kernel_size=3,
                                    stride=1, padding=1, groups=num_filters, bias=True)

        self.attention = ScaledDotProductAttention(scaling_factor=feature_size ** 0.5)

    def forward(self, x):

        width = x.size(2)

        value = self.value_conv(x)

        x = x.view(x.size(0), x.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        residual = x
        query = self.query_linear(x)
        key = self.key_linear(x)

        # Transpose for attention dot product: b x n x lq x dv
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        output, attention = self.attention(query, key, value)
        output = output.transpose(1, 2)

        output += residual

        output = output.view(output.size(0), output.size(1), width, width)

        return output, attention


class ConvolutionalFeatureAttention(nn.Module):
    ''' Scaled Dot-Product Attention
                    "All you need is attention"
    '''

    def __init__(self, feature_size, num_filters=64, dropout=0.1):
        super().__init__()

        # self.query_conv = nn.Conv2d(num_filters, num_filters, kernel_size=5,
        #                           stride=1, padding=2, groups=num_filters, bias=True)
        self.query_cnn = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=5,
                      stride=1, padding=2, groups=num_filters, bias=True),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=5,
                      stride=1, padding=2, groups=num_filters, bias=True),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=5,
                      stride=1, padding=2, groups=num_filters, bias=True)
            )
        # self.key_conv = nn.Conv2d(num_filters, num_filters, kernel_size=5,
        #                         stride=1, padding=2, groups=num_filters, bias=True)
        self.key_cnn = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=5,
                      stride=1, padding=2, groups=num_filters, bias=True),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=5,
                      stride=1, padding=2, groups=num_filters, bias=True),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=5,
                      stride=1, padding=2, groups=num_filters, bias=True)
            )
        self.value_conv = nn.Conv2d(num_filters, num_filters, kernel_size=3,
                                    stride=1, padding=1, groups=num_filters, bias=True)

        self.attention = ScaledDotProductAttention(scaling_factor=feature_size ** 0.5)

    def forward(self, x):

        width = x.size(2)

        # key = self.key_conv(x)
        # query = self.query_conv(x)
        key = self.key_cnn(x)
        query = self.query_cnn(x)
        value = self.value_conv(x)

        x = x.view(x.size(0), x.size(1), -1)
        key = key.view(value.size(0), value.size(1), -1)
        query = query.view(value.size(0), value.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        residual = x

        # Transpose for attention dot product
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        output, attention = self.attention(query, key, value)
        output = output.transpose(1, 2)

        output += residual

        output = output.view(output.size(0), output.size(1), width, width)

        return output, attention


class SpatialAttention(nn.Module):

    def __init__(self, height, width, stride, num_channels):
        super().__init__()

        self.height = height
        self.width = width
        self.stride = stride
        self.num_channels = num_channels

        self.vanilla_attention = VanillaMLPAttention(
            feature_size=height*width*num_channels, encoding_size=16)

    def forward(self, x):

        y = torch.zeros_like(x)
        for i in range(x.size(2)//self.height):
            for j in range(x.size(3)//self.height):
                spatial_feature = x[:, :, i*self.stride:i*self.stride +
                                    self.height,  j*self.stride:j*self.stride+self.width]
                spatial_feature = spatial_feature.reshape(spatial_feature.size(0), -1)
                y[:, :, i*self.stride:i*self.stride+self.height,  j*self.stride:j*self.stride+self.width], attention = self.vanilla_attention(
                    spatial_feature).view(spatial_feature.size(0), self.num_channels, self.height, self.width)

        return y
