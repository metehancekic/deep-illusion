import os
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import torch

from models.resnet import ResNet
from models.lenet import CNN

from deepillusion.torchattacks.analysis import get_perturbation_stats


def save_image(model, test_loader, attack_params, attack_args, attack_func="PGD"):

    device = model.parameters().__next__().device
    data, target = iter(test_loader).__next__()

    data, target = data.to(device), target.to(device)

    perturbs = attack_func(x=data, y_true=target, **attack_args)

    data_adv = data + perturbs

    output = model(data_adv)

    pred = output.argmax(dim=1, keepdim=True)

    fig = plt.figure()
    plt.imshow(data.permute(0, 2, 3, 1).cpu().numpy()[0])
    plt.axis("off")
    plt.savefig(str(target.cpu().numpy()[0])+".pdf")

    fig = plt.figure()
    plt.imshow(data_adv.permute(0, 2, 3, 1).cpu().numpy()[0])
    plt.axis("off")
    plt.savefig(str(pred.cpu().numpy()[0])+".pdf")

    fig = plt.figure()
    plt.imshow(10 * (perturbs+attack_params["eps"]).permute(0, 2, 3, 1).cpu().numpy()[0])
    plt.axis("off")
    plt.savefig("perturb"+".pdf")


def initiate_cifar10(random_model=False):

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
    model_2 = ResNet().to(device)
    if not random_model:
        if not use_cuda:
            model.load_state_dict(torch.load(
                "checkpoints/cifar/resnet_NT_ep_100.pt", map_location='cpu'))
            model_2.load_state_dict(torch.load(
                "checkpoints/cifar/resnet_RFGSM_eps_8_a_10_ep_100.pt", map_location='cpu'))
        else:
            model.load_state_dict(torch.load("checkpoints/cifar/resnet_NT_ep_100.pt"))
            model_2.load_state_dict(torch.load(
                "checkpoints/cifar/resnet_RFGSM_eps_8_a_10_ep_100.pt"))
    model.eval()
    model_2.eval()
    test_loss, test_acc = test(model_2, test_loader)
    print(f'Clean \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    return model, model_2, train_loader, test_loader


def initiate_mnist(dataset, random_model=False):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if dataset == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                root='./data',
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
                ),
            batch_size=50,
            shuffle=True,
            **kwargs
            )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                './data',
                train=False,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
                ),
            batch_size=10000,
            shuffle=True,
            **kwargs
            )

    elif dataset == "fashion":
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                root='./data',
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
                ),
            batch_size=50,
            shuffle=True,
            **kwargs
            )

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                './data',
                train=False,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
                ),
            batch_size=10000,
            shuffle=True,
            **kwargs
            )
    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

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

    model = CNN().to(device)
    model_2 = CNN().to(device)
    if not random_model:
        if not use_cuda:
            model.load_state_dict(torch.load("checkpoints/" + dataset +
                                             "/CNN_adv_inf_0.3.pt", map_location='cpu'))
            model_2.load_state_dict(torch.load(
                "checkpoints/" + dataset + "/CNN.pt", map_location='cpu'))
        else:
            model.load_state_dict(torch.load("checkpoints/" + dataset + "/CNN_adv_inf_0.3.pt"))
            model_2.load_state_dict(torch.load("checkpoints/" + dataset + "/CNN.pt"))
        # model.load_state_dict(torch.load("checkpoints/CNN.pt"))
        model.eval()
        model_2.eval()
        test_loss, test_acc = test(model, test_loader)
        print(f'Clean \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    return model, model_2, train_loader, test_loader


def test_adversarial(model, test_loader, attack_params, attack_args, attack_func="PGD"):

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

        perturbs = attack_func(x=data, y_true=target, **attack_args)

        data_adv = data + perturbs
        perturbation_properties = get_perturbation_stats(
            data, data_adv, attack_params["eps"], verbose=True)

        output = model(data_adv)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data_adv.size(0)

        pred = output.argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)

    return test_loss/test_size, test_correct/test_size


def test_targeted_adversarial(model, test_loader, attack_params, attack_args, attack_func="PGD"):

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

        perturbs = attack_func(x=data, y_target=(target-1) % 10, **attack_args)

        data_adv = data + perturbs
        perturbation_properties = get_perturbation_stats(
            data, data_adv, attack_params["eps"], verbose=True)

        output = model(data_adv)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data_adv.size(0)

        pred = output.argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)

    return test_loss/test_size, test_correct/test_size


def test(model, test_loader):

    model.eval()

    device = model.parameters().__next__().device

    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            cross_ent = nn.CrossEntropyLoss()
            test_loss += cross_ent(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
    test_size = len(test_loader.dataset)

    return test_loss/test_size, test_correct/test_size
