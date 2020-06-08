from tqdm import tqdm
from warnings import warn
import torch
from torch import nn

from .._utils import GradientMaskingWarning, GradientMaskingError
from ._utils import clip, to_one_hot

__all__ = ["cw_single_step_grad", "cw_single_step_sign", "CWlinf", "CWlinf_e"]


def cw_single_step_grad(net, x, y_true, num_classes=10, verbose=False):
    """
    Description: Single Step
    Input :
        net : Neural Network                                        (torch.nn.Module)
        x : Inputs to the net                                       (Batch)
        y_true : Labels                                             (Batch)
        data_params :                                               (dict)
            x_min:  Minimum possible value of x (min pixel value)   (Float)
            x_max:  Maximum possible value of x (max pixel value)   (Float)
        attack_params : Attack parameters as a dictionary           (dict)
        verbose: Check for gradient masking                         (Bool)
    Output:
        perturbation : Single step perturbation (Clamped with input limits)

    Explanation:
        e = grad_{x}(net(x)_true - net(x)_target)
    """
    e = torch.zeros_like(x, requires_grad=True)  # perturbation

    # Increase precision to prevent gradient masking
    if x.device.type == "cuda":
        y_hat = net(x + e).type(torch.cuda.DoubleTensor)
    else:
        y_hat = net(x + e)

    y_true_onehot = to_one_hot(y_true, num_classes).to(x.device)

    correct_logit = (y_true_onehot * y_hat).sum(dim=1)
    wrong_logit = ((1 - y_true_onehot) * y_hat - 1e4 * y_true_onehot).max(dim=1)[0]

    loss = -nn.functional.relu(correct_logit - wrong_logit + 50)

    # Calculating backprop for images
    loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float), retain_graph=True)
    e_grad = e.grad.data

    if verbose:
        # To make sure Gradient Masking is not happening
        max_attack_for_each_image, _ = e_grad.abs().view(e.size(0), -1).max(dim=1)
        if max_attack_for_each_image.min() <= 0:
            warn("Gradient Masking is happening for some images!!!!!", GradientMaskingWarning)

    perturbation = e_grad

    return perturbation


def cw_single_step_sign(net, x, y_true, data_params, attack_params, num_classes=10, verbose=False):
    """
    Description: Single Step
    Input :
        net : Neural Network                                        (torch.nn.Module)
        x : Inputs to the net                                       (Batch)
        y_true : Labels                                             (Batch)
        data_params :                                               (dict)
            x_min:  Minimum possible value of x (min pixel value)   (Float)
            x_max:  Maximum possible value of x (max pixel value)   (Float)
        attack_params : Attack parameters as a dictionary           (dict)
        verbose: Check for gradient masking                         (Bool)
    Output:
        perturbation : Single step perturbation (Clamped with input limits)

    Explanation:
        e = epsilon * sign(grad_{x}(net(x)_true - net(x)_target))
    """
    e = torch.zeros_like(x, requires_grad=True)  # perturbation

    # Increase precision to prevent gradient masking
    if x.device.type == "cuda":
        y_hat = net(x + e).type(torch.cuda.DoubleTensor)
    else:
        y_hat = net(x + e)

    y_true_onehot = to_one_hot(y_true, num_classes).to(x.device)

    correct_logit = (y_true_onehot * y_hat).sum(dim=1)
    wrong_logit = ((1 - y_true_onehot) * y_hat - 1e4 * y_true_onehot).max(dim=1)[0]

    loss = -nn.functional.relu(correct_logit - wrong_logit + 50)

    # Calculating backprop for images
    loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float), retain_graph=True)
    e_grad = e.grad.data

    if verbose:
        # To make sure Gradient Masking is not happening
        max_attack_for_each_image, _ = e_grad.abs().view(e.size(0), -1).max(dim=1)
        if max_attack_for_each_image.min() <= 0:
            warn("Gradient Masking is happening for some images!!!!!", GradientMaskingWarning)

    if attack_params["norm"] == "inf":
        perturbation = attack_params["eps"] * e_grad.sign()
    else:
        perturbation = e_grad * attack_params["eps"] / \
            e_grad.view(e.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1)

    # Clipping perturbations so that  x_min < image + perturbation < x_max
    perturbation.data = clip(perturbation, data_params["x_min"] - x, data_params["x_max"] - x)
    return perturbation


def CWlinf(net, x, y_true, data_params, attack_params, verbose=False, progress_bar=False):
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
            e += delta * sign(grad_{x}(net(x)_true - net(x)_target)) 
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
            cw_single_step_sign_args = dict(net=net,
                                            x=x+perturbation,
                                            y_true=y_true,
                                            data_params=data_params,
                                            attack_params={"norm": attack_params["norm"],
                                                           "eps": attack_params["step_size"]},
                                            verbose=verbose)
            perturbation += cw_single_step_sign(**cw_single_step_sign_args)

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


def CWlinf_e(net, x, y_true, data_params, attack_params, verbose=False, progress_bar=False):
    """
    Description: Experctation Over Transformation
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
            grad_expectation = 0
            repeat ensemble_size:
                grad_expectation += grad_{x}(net(x)_true - net(x)_target)
            e += delta * sign(grad_expectation)
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
            cw_single_step_grad_args = dict(net=net,
                                            x=torch.clamp(x+perturbation,
                                                          data_params["x_min"],
                                                          data_params["x_max"]),
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
                expected_grad += cw_single_step_grad(**cw_single_step_grad_args)

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
