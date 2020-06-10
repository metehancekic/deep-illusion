"""
Description: Fast Gradient Sign Method
Goodfellow [https://arxiv.org/abs/1412.6572]

Example Use:

fgsm_args = dict(net=net,
                 x=x,
                 y_true=y_true,
                 data_params={"x_min": 0.,
                              "x_max": 1.},
                 attack_params={"norm": "inf",
                                "eps": 8.0/255}
                 optimizer=optimizer)
perturbs = FGSM(**fgsm_args)
data_adversarial = data + perturbs

fgsm_targeted_args = dict(net=net,
                          x=x,
                          y_target=y_target,
                          data_params={"x_min": 0.,
                                       "x_max": 1.},
                          attack_params={"norm": "inf",
                                         "eps": 8.0/255},
                          optimizer=optimizer)
perturbs = FGSM_targeted(**fgsm_targeted_args)
data_adversarial = data + perturbs

"""


from apex import amp
import torch
from torch import nn

from .._utils import clip


__all__ = ["FGSM", "FGM", "FGSM_targeted"]


def FGSM(net, x, y_true, data_params, attack_params, optimizer=None, verbose=False):
    """
    Description: Fast gradient sign method
        Goodfellow [https://arxiv.org/abs/1412.6572]
    Input :
        net : Neural Network                                        (torch.nn.Module)
        x : Inputs to the net                                       (Batch)
        y_true : Labels                                             (Batch)
        data_params :                                               (dict)
            x_min:  Minimum possible value of x (min pixel value)   (Float)
            x_max:  Maximum possible value of x (max pixel value)   (Float)
        attack_params : Attack parameters as a dictionary           (dict)
            norm : Norm of attack                                   (Str)
            eps : Attack budget                                     (Float)
        optimizer : Optimizer
        verbose : Verbosity
    Output:
        perturbation : Single step perturbation (Clamped with input limits)

    Explanation:
        e = epsilon * sign(grad_{x}(net(x)))
    """
    e = torch.zeros_like(x, requires_grad=True)

    y_hat = net(x + e).type(torch.cuda.DoubleTensor)

    # Loss computation
    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(y_hat, y_true)

    # Calculating backprop with amp for images
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward(gradient=torch.ones_like(
            y_true, dtype=torch.float), retain_graph=True)

    e_grad = e.grad.data
    if attack_params["norm"] == "inf":
        perturbation = attack_params["eps"] * e_grad.sign()
    else:
        perturbation = e_grad * attack_params["eps"] / \
            e_grad.view(e.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1)

    # Clipping perturbations so that  x_min < image + perturbation < x_max
    perturbation.data = clip(perturbation, data_params["x_min"] - x, data_params["x_max"] - x)
    assert (x+perturbation).min() >= 0 and (x+perturbation).max() <= 1
    return perturbation


def FGM(net, x, y_true, optimizer=None, verbose=False):
    """
    Description: Fast gradient method (without sign gives gradients as it is)
        Goodfellow [https://arxiv.org/abs/1412.6572]
    Input :
        net : Neural Network                                        (torch.nn.Module)
        x : Inputs to the net                                       (Batch)
        y_true : Labels                                             (Batch)
        data_params :                                               (dict)
            x_min:  Minimum possible value of x (min pixel value)   (Float)
            x_max:  Maximum possible value of x (max pixel value)   (Float)
        attack_params : Attack parameters as a dictionary           (dict)
        optimizer: Optimizer
        verbose: Check for gradient masking                         (Bool)
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

    # Calculating backprop with amp for images
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward(gradient=torch.ones_like(
            y_true, dtype=torch.float), retain_graph=True)

    e_grad = e.grad.data
    perturbation = e_grad

    return perturbation


def FGSM_targeted(net, x, y_target, data_params, attack_params, optimizer=None, verbose=False):
    """
    Description: Fast gradient sign method
        Goodfellow [https://arxiv.org/abs/1412.6572]
    Input :
        net : Neural Network                                        (torch.nn.Module)
        x : Inputs to the net                                       (Batch)
        y_target : Target label                                     (Batch)
        data_params :                                               (dict)
            x_min:  Minimum possible value of x (min pixel value)   (Float)
            x_max:  Maximum possible value of x (max pixel value)   (Float)
        attack_params : Attack parameters as a dictionary           (dict)
            norm : Norm of attack                                   (Str)
            eps : Attack budget                                     (Float)
    Output:
        perturbation : Single step perturbation (Clamped with input limits)

    Explanation:
        e = epsilon * sign(grad_{x}(net(x)))
    """
    e = torch.zeros_like(x, requires_grad=True)  # perturbation

    # Increase precision to prevent gradient masking
    y_hat = net(x + e).type(torch.cuda.DoubleTensor)

    # Loss computation
    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(y_hat, y_target)

    # Calculating backprop for images
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward(gradient=torch.ones_like(
            y_target, dtype=torch.float), retain_graph=True)

    e_grad = e.grad.data
    if attack_params["norm"] == "inf":
        perturbation = -attack_params["eps"] * e_grad.sign()
    else:
        perturbation = -e_grad * attack_params["eps"] / \
            e_grad.view(e.shape[0], -1).norm(p=attack_params["norm"], dim=-1).view(-1, 1, 1, 1)

    # Clipping perturbations so that  x_min < image + perturbation < x_max
    perturbation.data = clip(perturbation, data_params["x_min"] - x, data_params["x_max"] - x)
    return perturbation
