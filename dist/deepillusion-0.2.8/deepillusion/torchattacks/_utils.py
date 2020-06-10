"""
Torch Auxilary tools
"""

import torch
import numpy as np

__all__ = ["cross_entropy_one_hot", "clip", "to_one_hot"]


def cross_entropy_one_hot(y_hat, y, reduction=None):
    """
    Args:
         y_hat: Output of neural net        (Batch)
         y: Original label                  (Batch)
         reduction: Reduce size by func     (Str)
    """
    if reduction == "mean":
        return torch.mean(torch.sum(-y * torch.nn.LogSoftmax(y_hat), dim=1))
    elif reduction == "sum":
        return torch.sum(torch.sum(-y * torch.nn.LogSoftmax(y_hat), dim=1))
    else:
        return torch.sum(-y * torch.nn.LogSoftmax(y_hat), dim=1)


def clip(x, lower, upper):
    """
    Args:
    input:
        x: Input tensor                    (Batch)
        lower: lower bound                 (float or torch.Tensor)
        upper: upper bound                 (float or torch.Tensor)
    output:
        x: Clipped x                       (Batch)
    """
    if isinstance(lower, torch.Tensor) and isinstance(upper, torch.Tensor):
        x = torch.max(torch.min(x, upper), lower)
    elif isinstance(lower, (float, int)) and isinstance(upper, (float, int)):
        x = torch.clamp(x, min=lower, max=upper)
    else:
        raise ValueError("lower and upper should be same type (float, int, torch.Tensor)")

    return x


def to_one_hot(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]
