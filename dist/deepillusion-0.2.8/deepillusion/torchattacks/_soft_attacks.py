"""
Description: Attack to generate exact probability distributions

soft_attack_single_step_args = dict(net=model,
                                    x=x,
                                    y_soft_vector=y_soft_vector,
                                    data_params={"x_min": 0.,
                                                 "x_max": 1.},
                                    attack_params={"norm": "inf",
                                                   "eps": 8./255},
                                    verbose=False)
perturbs = soft_attack_single_step(**soft_attack_single_step_args)
data_adversarial = data + perturbs

iterative_soft_attack_args = dict(net=model,
                                    x=x,
                                    y_soft_vector=y_soft_vector,
                                    data_params={"x_min": 0.,
                                                 "x_max": 1.},
                                    attack_params={"norm": "inf",
                                                   "eps": 8./255,
                                                   "step_size": 2./255,
                                                   "num_steps": 7,
                                                   "random_start": False,
                                                   "num_restarts": 1},
                                    verbose=False)
perturbs = iterative_soft_attack(**iterative_soft_attack_args)
data_adversarial = data + perturbs
"""

from tqdm import tqdm
import torch
from torch import nn

from ._utils import cross_entropy_one_hot, clip

__all__ = ["soft_attack_single_step", "iterative_soft_attack"]


def soft_attack_single_step(net, x, y_soft_vector, data_params, attack_params, verbose=False):
    """
    Input :
        net : Neural Network (Classifier)
        x : Inputs to the net
        y_soft_vector : Labels
        data_params: Data parameters as dictionary
            x_min : Minimum legal value for elements of x
            x_max : Maximum legal value for elements of x
        attack_params : Attack parameters as a dictionary
            norm : Norm of attack
            eps : Attack budget
        verbose: Verbosity
    Output:
        perturbs : Perturbations for given batch
    """

    e = torch.zeros_like(x, requires_grad=True)

    if x.device.type == "cuda":
        y_hat = net(x + e).type(torch.cuda.DoubleTensor)
    else:
        y_hat = net(x + e)

    loss = cross_entropy_one_hot(y_hat, y_soft_vector)

    loss.backward(gradient=torch.ones(y_soft_vector.size(
        0), dtype=torch.float, device=y_soft_vector.device), retain_graph=True)

    e_grad = e.grad.data

    if attack_params["norm"] == "inf":
        perturbation = -attack_params["eps"] * e_grad
    else:
        perturbation = (e - e_grad * attack_params['eps']) / \
            (e - e_grad).view(e.shape[0], -
                              1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1)

    perturbation.data = clip(perturbation, data_params["x_min"] - x, data_params["x_max"] - x)
    assert (x+perturbation).min() >= 0 and (x+perturbation).max() <= 1
    return perturbation


def iterative_soft_attack(net, x, y_soft_vector, data_params, attack_params, verbose=False, progress_bar=False):
    """
    Input :
        net : Neural Network (Classifier)
        x : Inputs to the net
        y_soft_vector : Labels
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
        verbose: Verbosity
        progress_bar: Progress bar
    Output:
        perturbation : Perturbations for given batch
    """

    if attack_params["random_start"]:
        if attack_params["norm"] == "inf":
            perturbation = (2 * torch.rand_like(x) - 1) * attack_params["eps"]
        else:
            e = 2 * torch.rand_like(x) - 1
            perturbation = e * attack_params["eps"] / \
                e.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1)

    else:
        perturbation = torch.zeros_like(x, dtype=torch.float)

    if progress_bar:
        iters = tqdm(iterable=range(attack_params["num_steps"]),
                     unit="step",
                     leave=False)
    else:
        iters = range(attack_params["num_steps"])

    for _ in iters:
        soft_attack_single_step_args = dict(net=net,
                                            x=x+perturbation,
                                            y_soft_vector=y_soft_vector,
                                            data_params=data_params,
                                            attack_params={"norm": attack_params["norm"],
                                                           "eps": attack_params["step_size"]},
                                            verbose=verbose)
        perturbation += soft_attack_single_step(**soft_attack_single_step_args)

        if attack_params["norm"] == "inf":
            perturbation = torch.clamp(perturbation, -attack_params["eps"], attack_params["eps"])
        else:
            perturbation = (perturbation * attack_params["eps"] /
                            perturbation.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1))

        if verbose:
            out = torch.nn.functional.softmax(net(x+perturbation))
            print(out[0])

    return perturbation
