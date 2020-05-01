"""
Auxilary tools
"""

import torch
import numpy as np

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

def perturbation_properties(clean_data, adversarial_data, epsilon):

    if isinstance(clean_data, torch.Tensor):
        clean_data = clean_data.cpu().numpy()
        adversarial_data = adversarial_data.cpu().numpy()

    e = adversarial_data.reshape([adversarial_data.shape[0], -1]) - \
        clean_data.reshape([clean_data.shape[0], -1])

    print(f"Attack budget: {epsilon}")

    print(f"Percent of images perturbation is added: {100. * np.count_nonzero(np.max(np.abs(e),axis = 1)) / e.shape[0]} %")
    print(f"L_inf distance: {np.round(np.abs(e).max()*255)}")
    print(f"Avg magnitude: {np.abs(e).mean()*255:.2f}")

    tol = 1e-5

    num_eps = (((np.abs(e) < epsilon + tol) & (np.abs(e) > epsilon - tol)).sum(axis=1).mean())

    print(f"Percent of pixels with mag=eps: {100*num_eps/(adversarial_data.shape[1] * adversarial_data.shape[2]*adversarial_data.shape[3])}")

    return e
