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
from deepillusion.torchdefenses import adversarial_epoch, adversarial_test

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

parser.add_argument("-tra", "--tr_attack", type=str, default="PGD", metavar="dn/fgsm/pgd",
                    help="Attack method")
parser.add_argument("--attack", type=str, default="PGD", metavar="fgsm/pgd",
                    help="Attack method",
                    )

args = parser.parse_args()

if args.dataset == "cifar":
    model, train_loader, test_loader = initiate_cifar10()
    attack_params = {
        "norm": "inf",
        "eps": 8./255,
        "alpha": 10./255,
        "step_size": 2./255,
        "num_steps": 7,
        "random_start": True,
        "num_restarts": 1,
        }
    epochs = 20

elif args.dataset == "mnist":
    model, train_loader, test_loader = initiate_mnist(args.dataset)
    attack_params = {
        "norm": "inf",
        "eps": 0.3,
        "alpha": 0.4,
        "step_size": 0.01,
        "num_steps": 40,
        "random_start": True,
        "num_restarts": 1,
        }
    epochs = 20

elif args.dataset == "fashion":
    model, train_loader, test_loader = initiate_mnist(args.dataset)
    attack_params = {
        "norm": "inf",
        "eps": 0.1,
        "alpha": 0.125,
        "step_size": 0.01,
        "num_steps": 40,
        "random_start": True,
        "num_restarts": 1,
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
checkpoint_name = args.model
if adversarial_args["attack"]:
    for key in attack_params:
        checkpoint_name += "_" + str(key) + "_" + str(attack_params[key])
checkpoint_name += ".pt"


for epoch in range(1, args.epochs + 1):

    start_time = time.time()
    train_args = dict(model=model,
                      train_loader=train_loader,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      adversarial_args=adversarial_args)
    train_loss, train_acc = adversarial_epoch(**train_args)

    test_args = dict(model=model,
                     test_loader=test_loader)
    test_loss, test_acc = adversarial_test(**test_args)
    end_time = time.time()

    print(f'{args.tr_attack} train \t loss: {train_loss:.4f} \t acc: {train_acc:.4f}\n')
    print(f'Test \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}\n')
    # print(f'{end_time - start time} seconds')


if not os.path.exists("./checkpoints/"):
    os.makedirs("./checkpoints/")
torch.save(model.state_dict(), args.directory + "checkpoints/" + checkpoint_name)


attack_params["random_start"] = False

adversarial_args = dict(attack=attacks[args.attack],
                        attack_args=dict(net=model,
                                         data_params=data_params,
                                         attack_params=attack_params,
                                         verbose=True))

for key in attack_params:
    print(key + ': ' + str(attack_params[key]))

test_args = dict(model=model,
                 test_loader=test_loader,
                 adversarial_args=adversarial_args,
                 verbose=True)
test_loss, test_acc = adversarial_test(**test_args)

print(f'{args.attack} test \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}\n')
