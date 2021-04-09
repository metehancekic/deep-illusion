"""
Description: Simultaneous Perturbation Stochastic Approximation

Example Use:


"""

from tqdm import tqdm
import torch
from torch import nn

from ._fgsm import FGSM, FGM
from ._utils import clip

__all__ = ["SPSA"]


def SPSA(net, x, y_true, data_params, attack_params, loss_function="cross_entropy", verbose=False, progress_bar=False):
    """
    Description: Basic Iterative Method
    Input :
                    net : Neural Network                                        (torch.nn.Module)
                    x : Inputs to the net                                       (Batch)
                    y_true : Labels                                             (Batch)
                    data_params :                                               (dict)
                                    x_min:  Minimum possible value of x (min pixel value)   (Float)
                                    x_max:  Maximum possible value of x (max pixel value)   (Float)
                    attack_params : Attack parameters as a dictionary           (dict)
                                                    norm : Norm of attack                               (Str)
                                                    eps : Attack budget                                 (Float)
                                                    step_size : Attack budget for each iteration        (Float)
                                                    num_steps : Number of iterations                    (Int)
                    verbose: check gradient masking                             (Bool)
                    progress_bar: Put progress bar                              (Bool)
    Output:
                    perturbation : Perturbations for given batch                (Batch)

    Explanation:
                    e = zeros()
                    repeat num_steps:
                                    e += delta * sign(grad_{x}(loss(net(x))))
    """

    # setting parameters.requires_grad = False increases speed

    if y_true is not None and len(x) != len(y_true):
        raise ValueError(f"Number of inputs {len(x)} should match the number of labels {len(y_true)}")
    if y_true is None:
        y_true = torch.argmax(net(x), dim=1)

    # Loss computation
    criterion = nn.CrossEntropyLoss(reduction="none")

    perturbation = torch.zeros_like(x, dtype=torch.float)

    # Adding progress bar for iterations if progress_bar = True
    if progress_bar:
        iters = tqdm(
            iterable=range(attack_params["num_steps"]),
            desc="Attack Steps Progress",
            unit="step",
            leave=False)
    else:
        iters = range(attack_params["num_steps"])

    for _ in iters:

        with torch.no_grad():

            if progress_bar:
                samples = tqdm(
                    iterable=range(attack_params["num_samples"]),
                    desc="Vector Samples Progress",
                    unit="sample",
                    leave=False)
            else:
                samples = range(attack_params["num_samples"])

            perturbation += grad_approx

            # Clip perturbation if surpassed the norm bounds
            if attack_params["norm"] == "inf":
                perturbation = torch.clamp(
                    perturbation, -attack_params["eps"], attack_params["eps"])

            perturbation.data = clip(
                perturbation, data_params["x_min"] - x, data_params["x_max"] - x)

    return perturbation


def _spsa_gradient():

        # grad_approx = torch.zeros_like(x)
 #    for _ in samples:
 #        rand_vector = torch.sign(torch.randn_like(x))
 #        g = (criterion(net(x + perturbation + attack_params["eps"] * rand_vector), y_true) -
 #             criterion(net(x + perturbation - attack_params["eps"] * rand_vector), y_true)).view(-1, 1, 1, 1) * \
 #            rand_vector / (2 * attack_params["eps"])

 #        grad_approx += (attack_params["step_size"] / attack_params["num_samples"]) * g
    pass
