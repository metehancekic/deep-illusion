"""
Description: Projected Gradient Descent
Madry

Example Use:

pgd_args = dict(net=model,
                x=x,
                y_true=y_true,
                data_params={"x_min": 0.,
                             "x_max": 1.},
                attack_params={"norm": "inf",
                               "eps": 8./255,
                               "step_size": 2./255,
                               "num_steps": 7,
                               "random_start": False,
                               "num_restarts": 1},
                verbose=False)
perturbs = PGD(**pgd_args)
data_adversarial = data + perturbs

"""

from tqdm import tqdm
import torch

from ._fgsm import FGSM

__all__ = ["PGD", "ensemble_PGD"]


def PGD(net, x, y_true, data_params, attack_params, verbose=True, warn_gradient_masking=False):
    """
    Description: Projected Gradient Descent
        Madry et al
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
                random_start : Randomly initialize image with perturbation  (Bool)
                num_restarts : Number of restarts                           (Int)
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

    perturbs = torch.zeros_like(x)

    # Adding progress bar for random-restarts if verbose = True
    if verbose and attack_params["num_restarts"] > 1:
        restarts = tqdm(
            iterable=range(attack_params["num_restarts"]),
            unit="restart",
            leave=True)
    else:
        restarts = range(attack_params["num_restarts"])

    for i in restarts:

        # Randomly initialize perturbation if needed
        if attack_params["random_start"] or attack_params["num_restarts"] > 1:
            if attack_params["norm"] == "inf":
                perturb = (2 * torch.rand_like(x) - 1) * attack_params["eps"]
            else:
                e = 2 * torch.rand_like(x) - 1
                perturb = e * attack_params["eps"] / \
                    e.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1)

        else:
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
            fgsm_args = dict(net=net,
                             x=x+perturb,
                             y_true=y_true,
                             data_params=data_params,
                             attack_params={"norm": attack_params["norm"],
                                            "eps": attack_params["step_size"]},
                             warn_gradient_masking=warn_gradient_masking)
            perturb += FGSM(**fgsm_args)

            # Clip perturbation if surpassed the norm bounds
            if attack_params["norm"] == "inf":
                perturb = torch.clamp(perturb, -attack_params["eps"], attack_params["eps"])
            else:
                perturb = (perturb * attack_params["eps"] /
                           perturb.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1))

        # Use the best perturbations among all restarts which fooled neural network
        if i == 0:
            perturbs = perturb.data
        else:
            output = net(torch.clamp(x + perturb, data_params["x_min"], data_params["x_max"]))
            y_hat = output.argmax(dim=1, keepdim=True)

            fooled_indices = (y_hat != y_true.view_as(y_hat)).nonzero()
            perturbs[fooled_indices] = perturb[fooled_indices].data

    # set back to True
    for p in net.parameters():
        p.requires_grad = True

    return perturbs


def ensemble_PGD(net, x, y_true, data_params, attack_params, ensemble_size=10, verbose=True):
    """
    Description: Projected Gradient Descent
        Madry et al
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
                random_start : Randomly initialize image with perturbation  (Bool)
                num_restarts : Number of restarts                           (Int)
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

    perturbs = torch.zeros_like(x)

    # Adding progress bar for random-restarts if verbose = True
    if verbose and attack_params["num_restarts"] > 1:
        restarts = tqdm(
            iterable=range(attack_params["num_restarts"]),
            unit="restart",
            leave=True)
    else:
        restarts = range(attack_params["num_restarts"])

    for i in restarts:
        # Randomly initialize perturbation if needed
        if attack_params["random_start"] or attack_params["num_restarts"] > 1:
            if attack_params["norm"] == "inf":
                perturb = (2 * torch.rand_like(x) - 1) * attack_params["eps"]
            else:
                e = 2 * torch.rand_like(x) - 1
                perturb = e * attack_params["eps"] / \
                    e.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1)

        else:
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
            fgsm_args = dict(net=net,
                             x=x+perturb,
                             y_true=y_true,
                             data_params=data_params,
                             attack_params={"norm": attack_params["norm"],
                                            "eps": attack_params["step_size"]},
                             warn_gradient_masking=warn_gradient_masking)

            for _ in range(ensemble_size):
                perturb += FGSM(**fgsm_args) / ensemble_size

            # Clip perturbation if surpassed the norm bounds
            if attack_params["norm"] == "inf":
                perturb = torch.clamp(perturb, -attack_params["eps"], attack_params["eps"])
            else:
                perturb = (perturb * attack_params["eps"] /
                           perturb.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1))

        # Use the best perturbations among all restarts which fooled neural network
        if i == 0:
            perturbs = perturb.data
        else:
            output = net(torch.clamp(x + perturb, data_params["x_min"], data_params["x_max"]))
            y_hat = output.argmax(dim=1, keepdim=True)

            fooled_indices = (y_hat != y_true.view_as(y_hat)).nonzero()
            perturbs[fooled_indices] = perturb[fooled_indices].data

    # set back to True
    for p in net.parameters():
        p.requires_grad = True

    return perturbs
