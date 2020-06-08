"""

"""


from __future__ import print_function

import numpy as np
import time
import argparse
import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


from deepillusion.torchattacks import FGSM, FGSM_targeted, RFGSM, PGD, ensemble_PGD, BIM, soft_attack_single_step, iterative_soft_attack
from deepillusion.torchattacks.analysis import whitebox_test
from deepillusion.torchdefenses import adversarial_epoch

from test_utils import initiate_cifar10, initiate_mnist, test_adversarial, save_image


parser = argparse.ArgumentParser(description='Test module')

# Dataset
parser.add_argument('--dataset', type=str, default='mnist', choices=[
                    "mnist", "fashion", "cifar"], metavar='mnist/fashion/cifar', help='Which dataset to use (default: mnist)')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.2,
                    metavar='LR', help='learning rate (default: 0.01)')

parser.add_argument("--lr_min", type=float, default=0., metavar="LR",
                    help="Learning rate min")

parser.add_argument("--lr_max", type=float, default=0.05, metavar="LR",
                    help="Learning rate max")

parser.add_argument('--momentum', type=float, default=0.5,
                    metavar='M', help='SGD momentum (default: 0.5)')

parser.add_argument('--weight_decay', type=float, default=0.0005,
                    metavar='WD', help='Weight decay (default: 0.0005)')

parser.add_argument("-tra", "--tr_attack", type=str, default="RFGSM", metavar="rfgsm/pgd",
                    help="Attack method")

parser.add_argument('--attack_method', type=str, default='PGD', choices=[
                    "FGSM", "FGSM_targeted", "RFGSM", "BIM", "BIM_EOT", "PGD", "PGD_EOT", "PGD_EOT_normalized", "PGD_EOT_sign"], metavar='', help='Attack method')

parser.add_argument('--loss_function', type=str, default='cross_entropy', choices=[
                    "cross_entropy", "carlini_wagner"], metavar='', help='Loss function to be used for attack')

args = parser.parse_args()

if args.dataset == "cifar":
    model, model_2, train_loader, test_loader = initiate_cifar10(random_model=True)
    attack_params = {
        "norm": "inf",
        "eps": 8./255,
        "alpha": 10./255,
        "step_size": 2./255,
        "num_steps": 7,
        "random_start": False,
        "num_restarts": 1,
        "EOT_size": 20,
        }
    epochs = 20

elif args.dataset == "mnist":
    model, model_2, train_loader, test_loader = initiate_mnist(args.dataset, random_model=True)
    attack_params = {
        "norm": "inf",
        "eps": 0.3,
        "alpha": 0.4,
        "step_size": 0.01,
        "num_steps": 40,
        "random_start": False,
        "num_restarts": 10,
        "EOT_size": 20,
        }
    epochs = 20

elif args.dataset == "fashion":
    model, model_2, train_loader, test_loader = initiate_mnist(args.dataset, random_model=True)
    attack_params = {
        "norm": "inf",
        "eps": 0.1,
        "alpha": 0.125,
        "step_size": 0.004,
        "num_steps": 40,
        "random_start": False,
        "num_restarts": 1,
        "EOT_size": 20,
        }
    epochs = 20

optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum,
                      weight_decay=args.weight_decay)
lr_steps = epochs * len(train_loader)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min,
                                              max_lr=args.lr_max, step_size_up=lr_steps/2,
                                              step_size_down=lr_steps/2)

attacks = dict(Standard=None,
               PGD=PGD,
               FGSM=FGSM,
               RFGSM=RFGSM)

data_params = {"x_min": 0., "x_max": 1.}

adversarial_args = dict(attack=attacks[args.tr_attack],
                        attack_args=dict(net=model,
                                         data_params=data_params,
                                         attack_params=attack_params,
                                         verbose=False))

# Checkpoint Namer
checkpoint_name = args.dataset + "_" + \
    str(adversarial_args["attack"].__name__) + "_e_" + str(epochs) + ".pt"


for epoch in range(1, epochs + 1):

    start_time = time.time()
    train_args = dict(model=model,
                      train_loader=train_loader,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      adversarial_args=adversarial_args)
    train_loss, train_acc = adversarial_epoch(**train_args)

    test_args = dict(model=model,
                     test_loader=test_loader)
    test_loss, test_acc = whitebox_test(**test_args)
    end_time = time.time()

    print(f'{args.tr_attack} train \t loss: {train_loss:.4f} \t acc: {train_acc:.4f}\t duration {end_time-start_time:.1f} seconds')
    print(f'Test \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}\n')


if not os.path.exists("./checkpoints/"):
    os.makedirs("./checkpoints/")
torch.save(model.state_dict(), "./checkpoints/" + checkpoint_name)


attack_params["random_start"] = False

attack_args = dict(data_params=data_params,
                   attack_params=attack_params,
                   loss_function=args.loss_function,
                   verbose=False)

if args.attack_method not in ["FGSM", "FGSM_targeted", "RFGSM"]:
    attack_args["progress_bar"] = True

adversarial_args = dict(attack=attacks[args.attack_method],
                        attack_args=attack_args)


whitebox_test_args = dict(model=model,
                          test_loader=test_loader,
                          adversarial_args=adversarial_args,
                          verbose=True,
                          progress_bar=True)

attack_loss, attack_acc = whitebox_test(**whitebox_test_args)
