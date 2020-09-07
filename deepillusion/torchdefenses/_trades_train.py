"""
Description: Training and testing functions for neural models with trades losse

functions:
    train: Performs a single training epoch (if attack_args is present adversarial training)
    test: Evaluates model by computing accuracy (if attack_args is present adversarial testing)
"""
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

__all__ = ["trades_loss", "trades_epoch"]


def trades_loss(model,
                x,
                y_true,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                num_steps=10,
                beta=1.0,
                norm='l_inf'):

    criterion_kl = nn.KLDivLoss(reduction="sum")
    model.eval()
    batch_size = len(x)
    # generate adversarial example
    x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()
    if norm == 'l_inf':
        for _ in range(num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif norm == 'l_2':
        delta = 0.001 * torch.randn(x.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / num_steps * 2)

        for _ in range(num_steps):
            adv = x + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x)
            delta.data.clamp_(0, 1).sub_(x)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()

    logits = model(x)
    loss_natural = F.cross_entropy(logits, y_true)
    loss_boundary = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                      F.softmax(model(x), dim=1))
    loss = loss_natural + beta * loss_boundary
    return loss, logits


def trades_epoch(model, train_loader, optimizer, trades_args, scheduler=None, progress_bar=False):
    """
    Description: Single trades epoch,
    Input :
            model : Neural Network               (torch.nn.Module)
            train_loader : Data loader           (torch.utils.data.DataLoader)
            optimizer : Optimizer                (torch.nn.optimizer)
            trades_args :
                    step_size:
                    eps:
                    num_steps:
                    beta:
            scheduler: Scheduler (Optional)      (torch.optim.lr_scheduler.CyclicLR)
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

        optimizer.zero_grad()

        # calculate trades loss
        loss, output = trades_loss(model=model,
                                   x=data,
                                   y_true=target,
                                   optimizer=optimizer,
                                   step_size=trades_args["step_size"],
                                   epsilon=trades_args["eps"],
                                   num_steps=trades_args["num_steps"],
                                   beta=trades_args["beta"])
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        train_loss += loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=False)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    train_size = len(train_loader.dataset)

    return train_loss/train_size, train_correct/train_size
