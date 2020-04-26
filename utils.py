"""
Auxilary tools
"""

import torch

class GradientMaskingError(ValueError):
    """Gradient masking error (false sense of robustness)"""

    def __init__(self, arg):
        super(GradientMaskingError, self).__init__()
        self.arg = arg


def cross_entropy_one_hot(y_hat, y, reduction=None):
    """ 
    Args:
         y_hat: Output of neural net	(Batch)
         y: Original label				(Batch)
         reduction: Reduce size by func	(Str)
    """
    if reduction == "mean":
        return torch.mean(torch.sum(-y * torch.nn.LogSoftmax(y_hat), dim=1))
    elif reduction == "sum":
        return torch.sum(torch.sum(-y * torch.nn.LogSoftmax(y_hat), dim=1))
    else:
        return torch.sum(-y * torch.nn.LogSoftmax(y_hat), dim=1)
