"""
Description: Random Fast Gradient Sign Method

Example Use:

rfgsm_args = dict(net=net,
                 x=x,
                 y_true=y_true,
                 data_params={"x_min": 0.,
                              "x_max": 1.},
                 attack_params={"norm": "inf",
                                "eps": 8.0/255,
                                "alpha": 10.0/255})
perturbation = RFGSM(**rfgsm_args)
"""

import torch
from torch import nn
from warnings import warn

from .._utils import GradientMaskingWarning, GradientMaskingError
from ._utils import clip, to_one_hot

__all__ = ["RFGSM"]


def RFGSM(net, x, y_true, data_params, attack_params, loss_function="cross_entropy", verbose=False):
    """
    Description: Random + Fast Gradient Sign Method
    Input :
        net : Neural Network            (torch.nn.Module)
        x : Inputs to the net           (Batch)
        y_true : Labels                 (Batch)
        data_params :
            x_min:  Minimum possible value of x (min pixel value)   (Float)
            x_max:  Maximum possible value of x (max pixel value)   (Float)
        attack_params :
            norm:   Attack norm         (Str)
            eps:    Attack budget       (Float)
            alpha:  Attack step size    (Float)
        verbose: Check for gradient masking                         (Bool)
    Output:
        perturbation : Single step perturbation (Clamped with input limits)

    Explanation:
        e = uniform(-eps,eps)
        e += alpha * sign(grad_{x}(net(x)))
        clip(e, -eps, eps)
    """
    if attack_params["norm"] == "inf":
        e = torch.rand_like(x) * 2 * attack_params['eps'] - attack_params['eps']
    else:
        e = 2 * torch.rand_like(x) - 1
        e = e * attack_params["eps"] / \
            e.view(x.shape[0], -1).norm(p=int(attack_params["norm"]), dim=-1).view(-1, 1, 1, 1)

    # Clip perturbation before gradient calculation to get best performance
    e.data = torch.max(torch.min(e, data_params["x_max"] - x), data_params["x_min"] - x)
    e.requires_grad = True

    y_hat = net(x + e)

    # Loss computation
    if loss_function == "cross_entropy":
        criterion = nn.CrossEntropyLoss(reduction="none")
        loss = criterion(y_hat, y_true)
    elif loss_function == "carlini_wagner":
        num_classes = y_hat.shape[-1]
        y_true_onehot = to_one_hot(y_true, num_classes).to(x.device)

        correct_logit = (y_true_onehot * y_hat).sum(dim=1)
        wrong_logit = ((1 - y_true_onehot) * y_hat - 1e4 * y_true_onehot).max(dim=1)[0]

        loss = -nn.functional.relu(correct_logit - wrong_logit + 50)
    else:
        raise NotImplementedError

    # Calculating backprop for images
    loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float), retain_graph=True)
    e_grad = e.grad.data

    if verbose:
        # To make sure Gradient Masking is not happening
        max_attack_for_each_image, _ = e_grad.abs().view(e.size(0), -1).max(dim=1)
        if max_attack_for_each_image.min() <= 0:
            warn("Gradient Masking is happening for some images!!!!!", GradientMaskingWarning)

    if attack_params["norm"] == "inf":
        perturbation = torch.clamp(
            e + attack_params['alpha'] * e_grad.sign(), -attack_params['eps'], attack_params['eps'])
    else:
        perturbation = (e + e_grad * attack_params['alpha']) / \
            (e+e_grad).view(e.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1)

    # Clipping perturbations so that  x_min < image + perturbation < x_max
    perturbation = clip(
        perturbation, data_params["x_min"] - x, data_params["x_max"] - x).detach()
    assert (x+perturbation).min() >= 0 and (x+perturbation).max() <= 1
    return perturbation
