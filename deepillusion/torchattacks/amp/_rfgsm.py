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
                                "alpha": 10.0/255},
                 optimizer=optimizer)
perturbation = RFGSM(**rfgsm_args)
"""

from apex import amp
import torch
from torch import nn

from .._utils import clip

__all__ = ["RFGSM"]


def RFGSM(net, x, y_true, data_params, attack_params, optimizer=None, verbose=False):
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
        optimizer: Optimizer
        verbose: Verbosity
    Output:
        perturbation : Single step perturbation (Clamped with input limits)
    """

    # Random perturbation initialization
    if attack_params["norm"] == "inf":
        e = torch.rand_like(x) * 2 * attack_params['eps'] - attack_params['eps']
    else:
        e = 2 * torch.rand_like(x) - 1
        e = e * attack_params["eps"] / \
            e.view(x.shape[0], -1).norm(p=int(attack_params["norm"]), dim=-1).view(-1, 1, 1, 1)

    # Clip randomly generated perturbation
    e.data = torch.max(torch.min(e, data_params["x_max"] - x), data_params["x_min"] - x)
    e.requires_grad = True

    y_hat = net(x + e)
    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(y_hat, y_true)

    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward(gradient=torch.ones_like(
            y_true, dtype=torch.float), retain_graph=True)

    e_grad = e.grad.data

    if attack_params["norm"] == "inf":
        perturbation = torch.clamp(
            e + attack_params['alpha'] * e_grad.sign(), -attack_params['eps'], attack_params['eps'])
    else:
        perturbation = (e + e_grad * attack_params['alpha']) / \
            (e+e_grad).view(e.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1)

    perturbation.data = clip(
        perturbation, data_params["x_min"] - x, data_params["x_max"] - x).detach()
    assert (x+perturbation).min() >= 0 and (x+perturbation).max() <= 1

    return perturbation
