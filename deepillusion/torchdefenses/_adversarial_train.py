"""
Description: Training and testing functions for neural models

functions:
    adversarial_epoch: Performs a single training epoch (if attack_args is present adversarial training)
    adversarial_test: Evaluates model by computing accuracy (if attack_args is present adversarial testing)
"""

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..torchattacks.analysis._evaluate import whitebox_test

__all__ = ['adversarial_epoch', 'adversarial_test']


def adversarial_epoch(model, train_loader, optimizer, scheduler=None, adversarial_args=None, progress_bar=False):
    """
    Description: Single epoch,
        if adversarial args are present then adversarial training.
    Input :
        model : Neural Network               (torch.nn.Module)
        train_loader : Data loader           (torch.utils.data.DataLoader)
        optimizer : Optimizer                (torch.nn.optimizer)
        scheduler: Scheduler (Optional)      (torch.optim.lr_scheduler.CyclicLR)
        adversarial_args :
            attack:                          (deepillusion.torchattacks)
            attack_args:
                attack arguments for given attack except "x" and "y_true"
        progress_bar:
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    model.train()

    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    if progress_bar:
        iter_train_loader = tqdm(
            iterable=train_loader,
            desc="Epoch Progress",
            unit="batch",
            leave=False)
    else:
        iter_train_loader = train_loader

    for data, target in iter_train_loader:

        data, target = data.to(device), target.to(device)

        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["net"] = model
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
            data += perturbs

        optimizer.zero_grad()
        output = model(data)
        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        train_loss += loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=False)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    train_size = len(train_loader.dataset)

    return train_loss/train_size, train_correct/train_size


def adversarial_test(model, test_loader, adversarial_args=None, verbose=False, progress_bar=False):
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

    return whitebox_test(model, test_loader, adversarial_args, verbose, progress_bar)
