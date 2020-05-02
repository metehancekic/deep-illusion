"""
Auxilary tools
"""

import torch
import numpy as np


__all__ = ["GradientMaskingError", "cross_entropy_one_hot", "get_perturbation_properties"]


class GradientMaskingError(ValueError):
    """Gradient masking error (false sense of robustness)"""

    def __init__(self, arg):
        super(GradientMaskingError, self).__init__()
        self.arg = arg


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


def get_perturbation_properties(clean_data, adversarial_data, epsilon, norm="inf", verbose=False):
    """
    Input:
        clean_data: Clean data                          (Batch)
        adversarial_data: Perturbed data                (Batch)
        epsilon: Attack Budget                          (Float)
    Output:
        perturbation_properties:                        (Dictionary)
            perturbation:
            percent_images_attacked:
            l_norm_distance:
            average_perturbation:
            percent_pixels_full_budget_perturbed:
    """
    if isinstance(clean_data, torch.Tensor):
        clean_data = clean_data.cpu().numpy()
        adversarial_data = adversarial_data.cpu().numpy()

    e = adversarial_data.reshape([adversarial_data.shape[0], -1]) - \
        clean_data.reshape([clean_data.shape[0], -1])

    percent_images_attacked = 100. * \
        np.count_nonzero(np.max(np.abs(e), axis=1)) / e.shape[0]
    if norm == "inf":
        l_norm_distance = np.round(np.abs(e).max()*255)
    else:
        raise NotImplementedError

    average_perturbation = np.abs(e).mean()*255

    tol = 1e-5
    num_eps = (((np.abs(e) < epsilon + tol) & (np.abs(e) > epsilon - tol)).sum(axis=1).mean())

    percent_pixels_full_budget_perturbed = 100 * num_eps/(e.shape[-1])

    if verbose:
        print(f"Attack budget: {epsilon}")
        print(f"Percent of images perturbation is added: {percent_images_attacked} %")
        print(f"L_inf distance: {l_norm_distance}")
        print(f"Avg magnitude: {average_perturbation:.2f}")
        print(f"Percent of pixels with mag=eps: {percent_pixels_full_budget_perturbed}")

    perturbation_properties = dict(perturbation=e,
                                   percent_images_attacked=percent_images_attacked,
                                   l_norm_distance=l_norm_distance,
                                   average_perturbation=average_perturbation,
                                   percent_pixels_full_budget_perturbed=percent_pixels_full_budget_perturbed)

    return perturbation_properties
