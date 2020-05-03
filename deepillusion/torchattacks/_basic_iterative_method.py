"""
Author: Metehan Cekic
Basic Iterative Method
"""

from tqdm import tqdm

import torch
import torchvision
from torch import nn

from ._fgsm import FGSM

__all__ = ["BIM"]


def BIM(net, x, y_true, data_params, attack_params, verbose=True):
    """
    Description: Basic Iterative Method
    Input :
        net : Neural Network            (torch.nn.Module)
        x : Inputs to the net           (Batch)
        y_true : Labels                 (Batch)
        verbose: Verbosity              (Bool)
        data_params :
            x_min:  Minimum possible value of x (min pixel value)   (Float)
            x_max:  Maximum possible value of x (max pixel value)   (Float)
        attack_params : Attack parameters as a dictionary
                norm : Norm of attack                               (Str)
                eps : Attack budget                                 (Float)
                step_size : Attack budget for each iteration        (Float)
                num_steps : Number of iterations                    (Int)
    Output:
        perturbs : Perturbations for given batch

    Explanation:
        e = zeros() or e = uniform(-eps,eps)
        repeat num_steps:
            e += delta * sign(grad_{x}(net(x)))
    """

    # setting parameters.requires_grad = False increases speed
    for p in net.parameters():
        p.requires_grad = False

    perturb = torch.zeros_like(x, dtype=torch.float)

    # Adding progress bar for iterations if verbose = True
    if verbose:
        iters = tqdm(
            iterable=range(attack_params["num_steps"]),
            unit="step",
            leave=True)
    else:
        iters = range(attack_params["num_steps"])

    for _ in iters:
        perturb += FGSM(net, x+perturb, y_true, attack_params["step_size"],
                        data_params, attack_params["norm"])

        # Clip perturbation if surpassed the norm bounds
        if attack_params["norm"] == "inf":
            perturb = torch.clamp(perturb, -attack_params["eps"], attack_params["eps"])
        else:
            perturb = (perturb * attack_params["eps"] /
                       perturb.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1))

    # set back to True
    for p in net.parameters():
        p.requires_grad = True

    return perturb
