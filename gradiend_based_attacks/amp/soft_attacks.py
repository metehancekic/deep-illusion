"""
Authors: Metehan Cekic
Date: 2020-04-23

Description: Attack code to aim soft predictions

Funcs: soft_attack_single_step
       iterative_soft_attack

"""

from tqdm import tqdm
from apex import amp

import torch
import torchvision
from torch import nn

from adversary.utils import cross_entropy_one_hot


def soft_attack_single_step(net, x, y_soft_vector, data_params, attack_params, optimizer=None):

    if attack_params["norm"] == "inf":
        e = torch.rand_like(x) * 2 * attack_params['eps'] - attack_params['eps']

    e = torch.zeros_like(x, requires_grad=True)

    if x.device.type == "cuda":
        y_hat = net(x + e).type(torch.cuda.DoubleTensor)
    else:
        y_hat = net(x + e).type(torch.DoubleTensor)

    loss = cross_entropy_one_hot(y_hat, y_soft_vector)

    loss.backward(gradient=torch.ones(y_soft_vector.size(
        0), dtype=torch.float, device=y_soft_vector.device), retain_graph=True)

    e_grad = e.grad.data

    if attack_params["norm"] == "inf":
        perturbation = -attack_params["step_size"] * e_grad
    else:
        perturbation = (e - e_grad * attack_params['eps']) / \
            (e - e_grad).view(e.shape[0], -
                              1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1)

    perturbation.data = torch.max(
        torch.min(perturbation, data_params["x_max"] - x), data_params["x_min"] - x)
    return perturbation


def iterative_soft_attack(net, x, y_soft_vector, data_params, attack_params, optimizer=None, verbose=False):
    """
    Input :
        net : Neural Network (Classifier)
        x : Inputs to the net
        y_true : Labels
        data_params: Data parameters as dictionary
                x_min : Minimum legal value for elements of x
                x_max : Maximum legal value for elements of x
        attack_params : Attack parameters as a dictionary
                norm : Norm of attack
                eps : Attack budget
                step_size : Attack budget for each iteration
                num_steps : Number of iterations
                random_start : Randomly initialize image with perturbation
                num_restarts : Number of restarts
    Output:
        perturbs : Perturbations for given batch
    """

    # fooled_indices = np.array(y_true.shape[0])
    perturbs = torch.zeros_like(x)

    if attack_params["random_start"]:
        if attack_params["norm"] == "inf":
            perturb = (2 * torch.rand_like(x) - 1) * attack_params["eps"]
        else:
            e = 2 * torch.rand_like(x) - 1
            perturb = e * attack_params["eps"] / \
                e.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1)

    else:
        perturb = torch.zeros_like(x, dtype=torch.float)

    if verbose:
        iters = tqdm(range(attack_params["num_steps"]))
    else:
        iters = range(attack_params["num_steps"])

    for _ in iters:
        perturb += soft_attack_single_step(net, torch.clamp(x+perturb, data_params["x_min"],
                                                            data_params["x_max"]),
                                           y_soft_vector, data_params, attack_params, optimizer)
        if attack_params["norm"] == "inf":
            perturb = torch.clamp(perturb, -attack_params["eps"], attack_params["eps"])
        else:
            perturb = (perturb * attack_params["eps"] /
                       perturb.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1))

        out = torch.nn.functional.softmax(net(x+perturb))
        print(out[0])

    perturbs = perturb.data

    perturbs.data = torch.max(
        torch.min(perturbs, data_params["x_max"] - x), data_params["x_min"] - x)

    return perturbs
