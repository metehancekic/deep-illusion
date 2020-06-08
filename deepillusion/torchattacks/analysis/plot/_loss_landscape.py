"""
Adversarial Evaluation functions for Pytorch neural models  
"""

from tqdm import tqdm

import torch
import torch.nn as nn
from .._fgsm import FGM

__all__ = ["loss_landscape"]


def loss_landscape(model, data_loader, adversarial_args, verbose=False, progress_bar=False):
    """
    Description: Evaluate model with test dataset,
            if adversarial args are present then adversarially perturbed test set.
    Input :
            model : Neural Network               (torch.nn.Module)
            test_loader : Data loader            (torch.utils.data.DataLoader)
            adversarial_args :                   (dict)
                    attack:                          (deepillusion.torchattacks)
                    attack_args:                     (dict)
                            attack arguments for given attack except "x" and "y_true"
            verbose: Verbosity                   (Bool)
            progress_bar: Progress bar           (Bool)
    Output:
            train_loss : Train loss              (float)
            train_accuracy : Train accuracy      (float)
    """
    device = model.parameters().__next__().device

    model.eval()
    breakpoint()

    for data, target in data_loader:

        data, target = data.to(device), target.to(device)

        fgm_args = dict(net=model,
                        x=data,
                        y_true=target,
                        verbose=verbose)

        grad = FGM(**fgm_args)
        grad /= grad.view(grad.shape[0], -1).norm(p=2, dim=-1).view(-1, 1, 1, 1)

        grad_single_image = grad[0]
        breakpoint()

        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss = cross_ent(output, target).item() * data.size(0)
