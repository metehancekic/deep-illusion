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
                verbose=False,
                progress_bar=False)
best_perturbation = PGD(**pgd_args)
data_adversarial = data + best_perturbation

"""

from tqdm import tqdm
import torch

from ._fgsm import FGSM, FGM

__all__ = ["PGD", "ePGD", "PEGD"]


def PGD(net, x, y_true, data_params, attack_params, verbose=False, progress_bar=False):
    """
    Description: Projected Gradient Descent
        Madry et al
    Input :
        net : Neural Network                (torch.nn.Module)
        x : Inputs to the net               (Batch)
        y_true : Labels                     (Batch)
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
        verbose: check gradient masking     (Bool)
        progress_bar: Put progress bar      (Bool)
    Output:
        best_perturbation : Perturbations for given batch

    Explanation:
        e = zeros() or e = uniform(-eps,eps)
        repeat num_steps:
            e += delta * sign(grad_{x}(net(x)))
    """

    # setting parameters.requires_grad = False increases speed
    for p in net.parameters():
        p.requires_grad = False

    best_perturbation = torch.zeros_like(x)

    # Adding progress bar for random-restarts if progress_bar = True
    if progress_bar and attack_params["num_restarts"] > 1:
        restarts = tqdm(
            iterable=range(attack_params["num_restarts"]),
            unit="restart",
            leave=False)
    else:
        restarts = range(attack_params["num_restarts"])

    for i in restarts:

        # Randomly initialize perturbation if needed
        if attack_params["random_start"] or attack_params["num_restarts"] > 1:
            if attack_params["norm"] == "inf":
                perturbation = (2 * torch.rand_like(x) - 1) * attack_params["eps"]
            else:
                e = 2 * torch.rand_like(x) - 1
                perturbation = e * attack_params["eps"] / \
                    e.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1)

        else:
            perturbation = torch.zeros_like(x, dtype=torch.float)

        # Adding progress bar for iterations if progress_bar = True
        if progress_bar:
            iters = tqdm(
                iterable=range(attack_params["num_steps"]),
                unit="step",
                leave=False)
        else:
            iters = range(attack_params["num_steps"])

        for _ in iters:
            fgsm_args = dict(net=net,
                             x=x+perturbation,
                             y_true=y_true,
                             data_params=data_params,
                             attack_params={"norm": attack_params["norm"],
                                            "eps": attack_params["step_size"]},
                             verbose=verbose)
            perturbation += FGSM(**fgsm_args)

            # Clip perturbation if surpassed the norm bounds
            if attack_params["norm"] == "inf":
                perturbation = torch.clamp(
                    perturbation, -attack_params["eps"], attack_params["eps"])
            else:
                perturbation = (perturbation * attack_params["eps"] /
                                perturbation.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1))

        # Use the best perturbations among all restarts which fooled neural network
        if i == 0:
            best_perturbation = perturbation.data
        else:
            output = net(torch.clamp(x + perturbation, data_params["x_min"], data_params["x_max"]))
            y_hat = output.argmax(dim=1, keepdim=True)

            fooled_indices = (y_hat != y_true.view_as(y_hat)).nonzero()
            best_perturbation[fooled_indices] = perturbation[fooled_indices].data

    # set back to True
    for p in net.parameters():
        p.requires_grad = True

    return best_perturbation


def ePGD(net, x, y_true, data_params, attack_params, verbose=False, progress_bar=False):
    """
    Description: Ensemble Projected Gradient Descent
        EOT paper
    Input :
        net : Neural Network            (torch.nn.Module)
        x : Inputs to the net           (Batch)
        y_true : Labels                 (Batch)
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
            ensemble_size: Ensemble size                                (Int)
        verbose: check gradient masking     (Bool)
        progress_bar: Put progress bar      (Bool)
    Output:
        best_perturbation : Perturbations for given batch

    Explanation:
        e = zeros() or e = uniform(-eps,eps)
        repeat num_steps:
            e += delta * sign(grad_{x}(net(x)))
    """

    # setting parameters.requires_grad = False increases speed
    for p in net.parameters():
        p.requires_grad = False

    best_perturbation = torch.zeros_like(x)

    # Adding progress bar for random-restarts if progress_bar = True
    if progress_bar and attack_params["num_restarts"] > 1:
        restarts = tqdm(
            iterable=range(attack_params["num_restarts"]),
            unit="restart",
            leave=False)
    else:
        restarts = range(attack_params["num_restarts"])

    for i in restarts:
        # Randomly initialize perturbation if needed
        if attack_params["random_start"] or attack_params["num_restarts"] > 1:
            if attack_params["norm"] == "inf":
                perturbation = (2 * torch.rand_like(x) - 1) * attack_params["eps"]
            else:
                e = 2 * torch.rand_like(x) - 1
                perturbation = e * attack_params["eps"] / \
                    e.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1)

        else:
            perturbation = torch.zeros_like(x, dtype=torch.float)

        # Adding progress bar for iterations if progress_bar = True
        if progress_bar:
            iters = tqdm(
                iterable=range(attack_params["num_steps"]),
                unit="step",
                leave=False)
        else:
            iters = range(attack_params["num_steps"])

        for _ in iters:
            fgsm_args = dict(net=net,
                             x=x+perturbation,
                             y_true=y_true,
                             data_params=data_params,
                             attack_params={"norm": attack_params["norm"],
                                            "eps": attack_params["step_size"]},
                             verbose=verbose)

            # Adding progress bar for ensemble if progress_bar = True
            if progress_bar:
                ensemble = tqdm(
                    iterable=range(attack_params["ensemble_size"]),
                    unit="element",
                    leave=False)
            else:
                ensemble = range(attack_params["ensemble_size"])
            for _ in ensemble:
                perturbation += FGSM(**fgsm_args) / attack_params["ensemble_size"]

            # Clip perturbation if surpassed the norm bounds
            if attack_params["norm"] == "inf":
                perturbation = torch.clamp(
                    perturbation, -attack_params["eps"], attack_params["eps"])
            else:
                perturbation = (perturbation * attack_params["eps"] /
                                perturbation.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1))

        # Use the best perturbations among all restarts which fooled neural network
        if i == 0:
            best_perturbation = perturbation.data
        else:
            output = net(torch.clamp(x + perturbation, data_params["x_min"], data_params["x_max"]))
            y_hat = output.argmax(dim=1, keepdim=True)

            fooled_indices = (y_hat != y_true.view_as(y_hat)).nonzero()
            best_perturbation[fooled_indices] = perturbation[fooled_indices].data

    # set back to True
    for p in net.parameters():
        p.requires_grad = True

    return best_perturbation


def PEGD(net, x, y_true, data_params, attack_params, verbose=False, progress_bar=False):
    """
    Description: Projected Expected Gradient Descent
        EOT paper
    Input :
        net : Neural Network            (torch.nn.Module)
        x : Inputs to the net           (Batch)
        y_true : Labels                 (Batch)
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
            ensemble_size: Ensemble size                                (Int)
        verbose: check gradient masking     (Bool)
        progress_bar: Put progress bar      (Bool)
    Output:
        best_perturbation : Perturbations for given batch

    Explanation:
        e = zeros() or e = uniform(-eps,eps)
        repeat num_steps:
            e += delta * sign(grad_{x}(net(x)))
    """

    # setting parameters.requires_grad = False increases speed
    for p in net.parameters():
        p.requires_grad = False

    best_perturbation = torch.zeros_like(x)

    # Adding progress bar for random-restarts if progress_bar = True
    if progress_bar and attack_params["num_restarts"] > 1:
        restarts = tqdm(
            iterable=range(attack_params["num_restarts"]),
            unit="restart",
            leave=False)
    else:
        restarts = range(attack_params["num_restarts"])

    for i in restarts:
        # Randomly initialize perturbation if needed
        if attack_params["random_start"] or attack_params["num_restarts"] > 1:
            if attack_params["norm"] == "inf":
                perturbation = (2 * torch.rand_like(x) - 1) * attack_params["eps"]
            else:
                perturbation = 2 * torch.rand_like(x) - 1
                perturbation = perturbation * attack_params["eps"] / \
                    perturbation.view(
                        x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1)

        else:
            perturbation = torch.zeros_like(x, dtype=torch.float)

        # Adding progress bar for iterations if progress_bar = True
        if progress_bar:
            iters = tqdm(
                iterable=range(attack_params["num_steps"]),
                unit="step",
                leave=False)
        else:
            iters = range(attack_params["num_steps"])

        for _ in iters:
            fgm_args = dict(net=net,
                            x=torch.clamp(x+perturbation,
                                          data_params["x_min"], data_params["x_max"]),
                            y_true=y_true,
                            verbose=verbose)

            # Adding progress bar for ensemble if progress_bar = True
            if progress_bar:
                ensemble = tqdm(
                    iterable=range(attack_params["ensemble_size"]),
                    unit="element",
                    leave=False)
            else:
                ensemble = range(attack_params["ensemble_size"])

            expected_grad = 0
            for _ in ensemble:
                e_grad = FGM(**fgm_args)
                e_grad = e_grad / e_grad.view(x.shape[0], -1).norm(p=2, dim=-1).view(-1, 1, 1, 1)
                expected_grad += e_grad

            # Clip perturbation if surpassed the norm bounds
            if attack_params["norm"] == "inf":
                perturbation += attack_params["step_size"] * expected_grad.sign()
                perturbation = torch.clamp(
                    perturbation, -attack_params["eps"], attack_params["eps"])
            else:
                perturbation += (expected_grad * attack_params["step_size"] /
                                 expected_grad.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1))
                perturbation = (perturbation * attack_params["eps"] /
                                perturbation.view(x.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1))

        # Use the best perturbations among all restarts which fooled neural network
        if i == 0:
            best_perturbation = perturbation.data
        else:
            output = net(torch.clamp(x + perturbation, data_params["x_min"], data_params["x_max"]))
            y_hat = output.argmax(dim=1, keepdim=True)

            fooled_indices = (y_hat != y_true.view_as(y_hat)).nonzero()
            best_perturbation[fooled_indices] = perturbation[fooled_indices].data

    # set back to True
    for p in net.parameters():
        p.requires_grad = True

    best_perturbation.data = clip(
        best_perturbation, data_params["x_min"] - x, data_params["x_max"] - x)
    return best_perturbation
