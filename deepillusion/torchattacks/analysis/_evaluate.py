"""
Adversarial Evaluation functions for Pytorch neural models  
"""

from tqdm import tqdm

import torch
import torch.nn as nn

__all__ = ["whitebox_test", "substitute_test", "save_adversarial_dataset"]


def whitebox_test(model, test_loader, adversarial_args=None, verbose=False, progress_bar=False):
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
            perturbs = adversarial_args['attack'](
                **adversarial_args["attack_args"])
            data += perturbs

        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)

    if verbose:
        print("#------ATTACK PARAMETERS------#")
        print("Attack Method: " + str(adversarial_args["attack"].__name__))
        print("\t" + "Loss Function: " +
              str(adversarial_args["attack_args"]["loss_function"]))
        for key in adversarial_args["attack_args"]["attack_params"]:
            print("\t" + key + ': ' +
                  str(adversarial_args["attack_args"]["attack_params"][key]))
        print(
            f'White-box test \t loss: {test_loss/test_size:.4f} \t acc: {test_correct/test_size:.4f}\n')

    return test_loss/test_size, test_correct/test_size


def substitute_test(model, substitute_model, test_loader, adversarial_args, verbose=False, progress_bar=False):
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
    assert device == model.parameters().__next__(
    ).device, "Model and substitute_model should be on same device"

    model.eval()
    substitute_model.eval()

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
            perturbs = adversarial_args['attack'](
                **adversarial_args["attack_args"])
            data += perturbs

        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)

    if verbose:
        print("#------ATTACK PARAMETERS------#")
        print("Attack Method: " + str(adversarial_args["attack"].__name__))
        print("\t" + "Loss Function: " +
              str(adversarial_args["attack_args"]["loss_function"]))
        for key in adversarial_args["attack_args"]["attack_params"]:
            print("\t" + key + ': ' +
                  str(adversarial_args["attack_args"]["attack_params"][key]))
        print(
            f'Substitute model black-box test \t loss: {test_loss/test_size:.4f} \t acc: {test_correct/test_size:.4f}\n')

    return test_loss/test_size, test_correct/test_size


def save_adversarial_dataset(model, test_loader, folder_dir="./", adversarial_args=None, verbose=False, progress_bar=False):
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
            test_loss : Test loss              (float)
            test_accuracy : Test accuracy      (float)
    """
    import numpy as np
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

    all_data = []
    all_adv = []
    all_label = []
    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        all_data.append(data.cpu().numpy())
        all_label.append(target.cpu().numpy())

        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["net"] = model
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](
                **adversarial_args["attack_args"])
            # breakpoint()
            data_adv = data + perturbs
            # print(f"max data_adv: {data_adv.max()}")
            # print(f"min data_adv: {data_adv.min()}")
            # print(f"max delta_adv: {(data_adv-data).max()}")
            # print(f"min delta_adv: {(data_adv-data).min()}")
            # print(f"max perturbs: {(perturbs).max()}")
            # print(f"min perturbs: {(perturbs).min()}")
            # print(f"max data: {data.max()}")
            # print(f"min data: {data.min()}")
        all_adv.append(data_adv.cpu().numpy())
        output = model(data_adv)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)

    np.save(folder_dir + "/adversarial_data.npy",
            np.concatenate(tuple(all_adv), axis=0))
    np.save(folder_dir + "/clean_data.npy",
            np.concatenate(tuple(all_data), axis=0))
    np.save(folder_dir + "/target.npy",
            np.concatenate(tuple(all_label), axis=0))

    if verbose:
        print("#------ATTACK PARAMETERS------#")
        print("Attack Method: " + str(adversarial_args["attack"].__name__))
        print("\t" + "Loss Function: " +
              str(adversarial_args["attack_args"]["loss_function"]))
        for key in adversarial_args["attack_args"]["attack_params"]:
            print("\t" + key + ': ' +
                  str(adversarial_args["attack_args"]["attack_params"][key]))
        print(
            f'White-box test \t loss: {test_loss/test_size:.4f} \t acc: {test_correct/test_size:.4f}\n')

    return test_loss/test_size, test_correct/test_size
