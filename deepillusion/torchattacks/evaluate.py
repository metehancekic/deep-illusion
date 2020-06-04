"""
Adversarial Evaluation functions for Pytorch neural models  
"""

from tqdm import tqdm

import torch
import torch.nn as nn

__all__ = ["adversarial_test", "black_box_test"]


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
    device = model.parameters().__next__().device

    model.eval()

    test_loss = 0
    test_correct = 0
    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            desc="Dataset Progress",
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["net"] = model
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
            data += perturbs

        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)

    if verbose:
        print("Attack Parameters:\n")
        for key in adversarial_args["attack_args"]["attack_params"]:
            print(key + ': ' + str(adversarial_args["attack_args"]["attack_params"][key]))
        print(f'Adversarial test \t loss: {test_loss/test_size:.4f} \t acc: {test_correct/test_size:.4f}\n')

    return test_loss/test_size, test_correct/test_size


def black_box_test(model, substitute_model, test_loader, adversarial_args, verbose=False, progress_bar=False):
    """
    Description: Evaluate model with black box attack generated from substitute model,
    Input :
            model : Neural Network for evaluation                                       (torch.nn.Module)
            substitute_model : Substitute Neural Network over attack is generated       (torch.nn.Module)
            test_loader : Data loader                                                   (torch.utils.data.DataLoader)
            adversarial_args :                                                          (dict)
                    attack:                                                             (deepillusion.torchattacks)
                    attack_args:                                                        (dict)
                            attack arguments for given attack except "x" and "y_true"
            verbose: Verbosity                                  (Bool)
            progress_bar: Progress bar                          (Bool)
    Output:
            train_loss : Train loss                             (float)
            train_accuracy : Train accuracy                     (float)
    """

    device = substitute_model.parameters().__next__().device
    assert device == model.parameters().__next__().device, "Model and substitute_model should be on same device"

    model.eval()

    test_loss = 0
    test_correct = 0
    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            desc="Dataset Progress",
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)
        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["net"] = substitute_model
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
            data += perturbs

        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)

    if verbose:
        print("Attack Parameters:\n")
        for key in adversarial_args["attack_args"]["attack_params"]:
            print(key + ': ' + str(adversarial_args["attack_args"]["attack_params"][key]))
        print(f'Black Box test \t loss: {test_loss/test_size:.4f} \t acc: {test_correct/test_size:.4f}\n')

    return test_loss/test_size, test_correct/test_size
