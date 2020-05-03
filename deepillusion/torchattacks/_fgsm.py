"""
Author: Metehan Cekic
Fast Gradient Sign Method
"""

import torch
import torchvision
from torch import nn

from .._utils import GradientMaskingError
from ._utils import clip

__all__ = ["FGSM", "FGSM_targeted"]


def FGSM(net, x, y_true, eps, data_params, norm="inf"):
    """
    Description: Fast gradient sign method
        Goodfellow [https://arxiv.org/abs/1412.6572]
    Input :
        net : Neural Network                                        (torch.nn.Module)
        x : Inputs to the net                                       (Batch)
        y_true : Labels                                             (Batch)
        eps:    Attack budget                                       (Float)
        data_params :
            x_min:  Minimum possible value of x (min pixel value)   (Float)
            x_max:  Maximum possible value of x (max pixel value)   (Float)
        norm:   Attack norm                                         (Str)
    Output:
        perturbation : Single step perturbation (Clamped with input limits)

    Explanation:
        e = epsilon * sign(grad_{x}(net(x)))
    """
    e = torch.zeros_like(x, requires_grad=True)  # perturbation

    # Increase precision to prevent gradient masking
    if x.device.type == "cuda":
        y_hat = net(x + e).type(torch.cuda.DoubleTensor)
    else:
        y_hat = net(x + e)

    # Loss computation
    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(y_hat, y_true)

    # Calculating backprop for images
    loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float), retain_graph=True)
    e_grad = e.grad.data
    # To make sure Gradient Masking is not happening
    max_attack_for_each_image, _ = e_grad.abs().view(e.size(0), -1).max(dim=1)
    if max_attack_for_each_image.min() <= 0:
        raise GradientMaskingError("Gradient masking is happening")

    if norm == "inf":
        perturbation = eps * e_grad.sign()
    else:
        perturbation = e_grad * eps / \
            e_grad.view(e.shape[0], -1).norm(p=norm, dim=-1).view(-1, 1, 1, 1)

    # Clipping perturbations so that  x_min < image + perturbation < x_max
    perturbation.data = clip(perturbation, data_params["x_min"] - x, data_params["x_max"] - x)
    assert (x+perturbation).min() >= 0 and (x+perturbation).max() <= 1
    return perturbation


def FGSM_targeted(net, x, y_target, eps, data_params, norm="inf"):
    """
    Description: Fast gradient sign method
        Goodfellow [https://arxiv.org/abs/1412.6572]
    Input :
        net : Neural Network                                        (torch.nn.Module)
        x : Inputs to the net                                       (Batch)
        y_target : Target label                                     (Batch)
        eps:    Attack budget                                       (Float)
        data_params :
            x_min:  Minimum possible value of x (min pixel value)   (Float)
            x_max:  Maximum possible value of x (max pixel value)   (Float)
        norm:   Attack norm                                         (Str)
    Output:
        perturbation : Single step perturbation (Clamped with input limits)

    Explanation:
        e = epsilon * sign(grad_{x}(net(x)))
    """
    e = torch.zeros_like(x, requires_grad=True)  # perturbation

    # Increase precision to prevent gradient masking
    if x.device.type == "cuda":
        y_hat = net(x + e).type(torch.cuda.DoubleTensor)
    else:
        y_hat = net(x + e)

    # Loss computation
    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(y_hat, y_target)

    # Calculating backprop for images
    loss.backward(gradient=torch.ones_like(y_target, dtype=torch.float), retain_graph=True)
    e_grad = e.grad.data
    # To make sure Gradient Masking is not happening
    max_attack_for_each_image, _ = e_grad.abs().view(e.size(0), -1).max(dim=1)
    if max_attack_for_each_image.min() <= 0:
        raise GradientMaskingError("Gradient masking is happening")

    if norm == "inf":
        perturbation = -eps * e_grad.sign()
    else:
        perturbation = -e_grad * eps / \
            e_grad.view(e.shape[0], -1).norm(p=norm, dim=-1).view(-1, 1, 1, 1)

    # Clipping perturbations so that  x_min < image + perturbation < x_max
    perturbation.data = clip(perturbation, data_params["x_min"] - x, data_params["x_max"] - x)
    assert (x+perturbation).min() >= 0 and (x+perturbation).max() <= 1

    return perturbation
