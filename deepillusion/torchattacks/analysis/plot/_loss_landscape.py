from ..._fgsm import FGSM, FGM
from ..._pgd import PGD
from . import _plot_settings
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
"""
Adversarial Evaluation functions for Pytorch neural models  
"""

from tqdm import tqdm
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib
from tqdm.contrib.itertools import product
matplotlib.use('Agg')


__all__ = ["loss_landscape"]


def loss_landscape(model, data_loader, data_params, attack_params, img_index=0, second_direction="adversarial", fig_name="loss_landscape.pdf", norm="inf", z_axis_type="loss", avg_points=1, verbose=False):
    """
    Description: Loss landscape plotting
    Input :
            model : Neural Network               (torch.nn.Module)
            data_loader : Data loader            (torch.utils.data.DataLoader)
            img_index : index in data loader     (int)
            second_direction : direction of axis (str)
            fig_name : figure name               (str)
            norm : Gradient norm                 (str="inf" or int>0)
            z_axis_type : Z axis                 (str)
            verbose: Verbosity                   (Bool)
    Output:
    """
    device = model.parameters().__next__().device

    model.eval()

    data, target = iter(data_loader).__next__()
    data, target = data.to(device), target.to(device)

    assert img_index >= 0 and img_index < len(data)

    single_image = data[img_index:img_index+1]
    single_label = target[img_index:img_index+1]

    # fgm_args = dict(net=model,
    #                 x=single_image,
    #                 y_true=single_label,
    #                 verbose=verbose)

    # grad = FGM(**fgm_args)

    pgd_args = dict(net=model,
                    x=single_image,
                    y_true=single_label,
                    data_params=data_params,
                    attack_params=attack_params,
                    verbose=verbose)

    grad = PGD(**pgd_args)

    if norm == "inf":
        grad = grad.sign()/255.
    elif type(norm).__name__ == "int" and norm >= 1:
        grad /= grad.view(grad.shape[0], -1).norm(p=2,
                                                  dim=-1).view(-1, 1, 1, 1)
    else:
        raise NotImplementedError

    first_axis = grad

    if second_direction == "adversarial":
        # fgm_args = dict(net=model,
        #                 x=single_image+first_axis,
        #                 y_true=single_label,
        #                 verbose=verbose)

        # grad_new = FGM(**fgm_args)

        pgd_args = dict(net=model,
                        x=single_image+first_axis,
                        y_true=single_label,
                        data_params=data_params,
                        attack_params=attack_params,
                        verbose=verbose)

        grad_new = PGD(**pgd_args)

        second_axis = grad_new

    elif second_direction == "random":
        second_axis = torch.randn_like(first_axis)

    if norm == "inf":
        second_axis = second_axis.sign()/255.
    elif type(norm).__name__ == "int" and norm >= 1:
        second_axis /= second_axis.view(
            second_axis.shape[0], -1).norm(p=norm, dim=-1)
    else:
        raise NotImplementedError

    # Keep orthogonal direction
    second_axis -= (second_axis*first_axis).sum() / \
        (first_axis*first_axis).sum() * first_axis

    axis_length = 100
    cross_ent = nn.CrossEntropyLoss()

    x_axis = np.linspace(-30, 30, axis_length)
    y_axis = np.linspace(-30, 30, axis_length)

    x = np.outer(x_axis, np.ones(axis_length))
    y = np.outer(y_axis, np.ones(axis_length)).T  # transpose
    z = np.zeros_like(x)

    colors = np.empty(x.shape, dtype=object)

    default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    default_color_cycle[7] = 'y'

    for i, j in tqdm(product(range(axis_length), range(axis_length))):
        current_point = (
            single_image + x_axis[i]*first_axis + y_axis[j]*second_axis)

        if avg_points > 1:
            images = current_point.repeat(avg_points, 1, 1, 1)

            noise = (torch.rand_like(images)*2-1)*0.5/255
            noise[0] = 0.0

            outputs = model(images+noise)

            output = outputs.mean(dim=0, keepdims=True)
        else:
            output = model(current_point)

        # output of single image misses batch size
        output = output.view(-1, output.shape[-1])
        test_loss = cross_ent(output, single_label).item()
        if z_axis_type == "loss":
            z[i, j] = test_loss
        elif z_axis_type == "confidence":
            preds = nn.functional.softmax(output, dim=1)
            prediction = preds.argmax()
            z[i, j] = preds[0, single_label[0]].detach().cpu().numpy()
            colors[i, j] = default_color_cycle[prediction]
        else:
            raise NotImplementedError

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if z_axis_type == "loss":
        ax.plot_surface(x, y, z, cmap='viridis', edgecolors='none')
        ax.set_title('Loss Landscape')
    elif z_axis_type == "confidence":
        ax.plot_surface(x, y, z, facecolors=colors,
                        edgecolors='none', shade=True)
        ax.set_title('Confidence Landscape')

    ax.set_xlim(np.min(x_axis), np.max(x_axis))
    ax.set_ylim(np.min(y_axis), np.max(y_axis))
    ax.set_xlabel(r'$g=sign(\nabla_{e}\mathcal{L})$')
    ax.set_ylabel(r'$g^{\perp}$')
    ax.set_zlabel(z_axis_type)
    plt.savefig(fig_name)
