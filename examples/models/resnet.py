
"""
Resnet Implementation
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class ResidualUnit(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, activate_before_residual=False):
        super(ResidualUnit, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.activate_before_residual = activate_before_residual
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

    def forward(self, x):

        if self.activate_before_residual:
            x = F.leaky_relu(self.bn1(x), 0.1)
        else:
            out = F.leaky_relu(self.bn1(x), 0.1)

        out = F.leaky_relu(self.bn2(self.conv1(x)), 0.1)
        out = self.conv2(out)

        if self.in_channel != self.out_channel:
            x = F.avg_pool2d(x, self.stride)
            x = F.pad(x, (0, 0, 0, 0, (self.out_channel - self.in_channel) //
                          2, (self.out_channel-self.in_channel)//2))

        out += x
        return out


class ResNet(nn.Module):

    def __init__(self, num_outputs=10):
        super(ResNet, self).__init__()

        filters = [16, 16, 32, 64]
        strides = [1, 2, 2]

        self.norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
            0.2471, 0.2435, 0.2616])

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = self.build_block(ResidualUnit, 5, filters[0], filters[1], strides[0], True)
        self.block2 = self.build_block(ResidualUnit, 5, filters[1], filters[2], strides[1], False)
        self.block3 = self.build_block(ResidualUnit, 5, filters[2], filters[3], strides[2], False)

        self.bn1 = nn.BatchNorm2d(filters[3])

        self.linear = nn.Linear(filters[-1], num_outputs)

    def build_block(self, unit, num_units, in_channel, out_channel, stride, activate_before_residual=False):
        layers = []
        layers.append(unit(in_channel, out_channel, stride, activate_before_residual))
        for i in range(num_units-1):
            layers.append(unit(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.norm(x)
        out = self.conv1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = F.leaky_relu(self.block3(out), 0.1)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class ResNetWide(nn.Module):

    def __init__(self, num_outputs=10):
        super(ResNetWide, self).__init__()

        filters = [16, 160, 320, 640]
        strides = [1, 2, 2]

        self.norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                              0.2471, 0.2435, 0.2616])
        # self.norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = self.build_block(ResidualUnit, 5, filters[0], filters[1], strides[0], True)
        self.block2 = self.build_block(ResidualUnit, 5, filters[1], filters[2], strides[1], False)
        self.block3 = self.build_block(ResidualUnit, 5, filters[2], filters[3], strides[2], False)

        self.bn1 = nn.BatchNorm2d(filters[3])

        self.linear = nn.Linear(filters[-1], num_outputs)

    def build_block(self, unit, num_units, in_channel, out_channel, stride, activate_before_residual=False):
        layers = []
        layers.append(unit(in_channel, out_channel, stride, activate_before_residual))
        for i in range(num_units-1):
            layers.append(unit(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.norm(x)
        out = self.conv1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = F.leaky_relu(self.block3(out), 0.1)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
