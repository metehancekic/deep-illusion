from ..._fgsm import FGSM, FGM
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
matplotlib.use('Agg')


__all__ = ["loss_landscape"]


def loss_landscape(model, data_loader, img_index=0, second_direction="adversarial", fig_name="loss_landscape.eps", norm="inf", z_axis_type="loss", verbose=False):
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

    single_image = data[img_index:img_index+1]
    single_label = target[img_index:img_index+1]

    fgm_args = dict(net=model,
                    x=single_image,
                    y_true=single_label,
                    verbose=verbose)

    grad = FGM(**fgm_args)

    if norm == "inf":
        grad = grad.sign()/255.
    elif type(norm).__name__ == "int" and norm >= 1:
        grad /= grad.view(grad.shape[0], -1).norm(p=2, dim=-1).view(-1, 1, 1, 1)
    else:
        raise NotImplementedError

    first_axis = grad

    if second_direction == "adversarial":
        fgm_args = dict(net=model,
                        x=single_image+first_axis,
                        y_true=single_label,
                        verbose=verbose)

        grad_new = FGM(**fgm_args)
        second_axis = grad_new

    elif second_direction == "random":
        second_axis = torch.randn_like(first_axis)

    if norm == "inf":
        second_axis = second_axis.sign()/255.
    elif type(norm).__name__ == "int" and norm >= 1:
        second_axis /= second_axis.view(second_axis.shape[0], -1).norm(p=norm, dim=-1)
    else:
        raise NotImplementedError

    # Keep orthogonal direction
    second_axis -= (second_axis*first_axis).sum() / (first_axis*first_axis).sum() * first_axis

    axis_length = 30
    cross_ent = nn.CrossEntropyLoss()

    x_axis = np.linspace(0, 4, axis_length)
    y_axis = np.linspace(0, 4, axis_length)

    x = np.outer(x_axis, np.ones(axis_length))
    y = np.outer(y_axis, np.ones(axis_length)).T  # transpose
    z = np.zeros_like(x)

    for i in range(axis_length):
        for j in range(axis_length):
            output = model(single_image + x_axis[i]*first_axis + y_axis[j]*second_axis)

            # output of single image misses batch size
            output = output.view(-1, output.shape[-1])
            test_loss = cross_ent(output, single_label).item()
            if z_axis_type == "loss":
                z[i, j] = test_loss
            elif z_axis_type == "confidence":
                preds = nn.functional.softmax(output)
                z[i, j] = preds[0, single_label[0]].detach().cpu().numpy()
            else:
                raise NotImplementedError

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
    ax.set_xlim(np.max(x), 0)
    ax.set_title('Loss Landscape')
    ax.set_xlabel(r'$g=sign(\nabla_{e}\mathcal{L})$')
    ax.set_ylabel(r'$g^{\perp}$')
    ax.set_zlabel(z_axis_type)
    plt.savefig(fig_name)
