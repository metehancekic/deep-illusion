
import time
from models.resnet import ResNet

from deepillusion.torchattacks import FGSM, RFGSM, PGD
from deepillusion.torchattacks.analysis import get_perturbation_stats

from torchvision import datasets, transforms

import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

transform = transforms.Compose([
    transforms.ToTensor(),
    ])

trainset = datasets.CIFAR10(root='./data',
                            train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset,
                                           batch_size=64,
                                           shuffle=True, **kwargs)

testset = datasets.CIFAR10(root='./data',
                           train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=200,
                                          shuffle=False, **kwargs)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def show_images(images, labels):
    num_img = len(images)
    np_images = [img.numpy() for img in images]
    fig, axes = plt.subplots(nrows=1, ncols=num_img, figsize=(20, 45))

    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        im = ax.imshow(np_images[i], vmin=0., vmax=1.)
        ax.set_title(f'{labels[i]}')
        plt.axis("off")

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.1, hspace=0.25)

    plt.show()


images, labels = iter(train_loader).next()
num_img_to_plot = 9
images = [images[i].permute(1, 2, 0) for i in range(num_img_to_plot)]
labels = [classes[i] for i in labels[:num_img_to_plot]]
# show_images(images, labels)

model = ResNet().to(device)
model.load_state_dict(torch.load("checkpoints/ResNet.pt"))
model.eval()


def test_adversarial(model, test_loader, data_params, attack_params):

    device = model.parameters().__next__().device

    for key in attack_params:
        print(key + ': ' + str(attack_params[key]))

    model.eval()
    test_loss = 0
    test_correct = 0
    test_load = tqdm(
        iterable=test_loader,
        unit="batch",
        leave=True)

    for data, target in test_load:

        data, target = data.to(device), target.to(device)

        # Attacks
        pgd_args = dict(net=model,
                        x=data,
                        y_true=target,
                        data_params=data_params,
                        attack_params=attack_params,
                        verbose=False)
        perturbs = PGD(**pgd_args)
        data_adv = data + perturbs
        perturbation_properties = get_perturbation_stats(
            data, data_adv, attack_params["eps"], verbose=False)

        output = model(data_adv)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data_adv.size(0)

        pred = output.argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)

    return test_loss/test_size, test_correct/test_size


data_params = {"x_min": 0., "x_max": 1.}
attack_params = {
    "norm": "inf",
    "eps": 8./255,
    "step_size": 2./255,
    "num_steps": 7,
    "random_start": False,
    "num_restarts": 1,
    }

start_time = time.time()
attack_loss, attack_acc = test_adversarial(model, test_loader,
                                           data_params=data_params,
                                           attack_params=attack_params)
print(f'Attack  \t loss: {attack_loss:.4f} \t acc: {attack_acc:.4f}')
print(f"{time.time()-start_time:.2f} seconds")
