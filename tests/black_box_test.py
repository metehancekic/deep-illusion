import time
import argparse

from deepillusion.torchattacks import FGSM, FGSM_targeted, RFGSM, PGD, ePGD, BIM, soft_attack_single_step, iterative_soft_attack
from deepillusion.torchattacks.analysis import whitebox_test, substitute_test

from test_utils import initiate_cifar10, initiate_mnist, save_image


parser = argparse.ArgumentParser(description='Test module')


# Dataset
parser.add_argument('--dataset', type=str, default='mnist', choices=[
                    "mnist", "fashion", "cifar"], metavar='mnist/fashion/cifar', help='Which dataset to use (default: mnist)')

args = parser.parse_args()

if args.dataset == "cifar":
    model, train_loader, test_loader = initiate_cifar10()
    attack_params = {
        "norm": "inf",
        "eps": 8./255,
        "alpha": 10./255,
        "step_size": 2./255,
        "num_steps": 7,
        "random_start": False,
        "num_restarts": 1,
        }
elif args.dataset == "mnist":
    model, model_2, train_loader, test_loader = initiate_mnist(args.dataset)
    attack_params = {
        "norm": "inf",
        "eps": 0.3,
        "alpha": 0.4,
        "step_size": 0.05,
        "num_steps": 10,
        "random_start": False,
        "num_restarts": 1,
        }

elif args.dataset == "fashion":
    model, train_loader, test_loader = initiate_mnist(args.dataset)
    attack_params = {
        "norm": "inf",
        "eps": 0.1,
        "alpha": 0.125,
        "step_size": 0.004,
        "num_steps": 100,
        "random_start": False,
        "num_restarts": 1,
        }

substitute_model = model_2

attack_params["ensemble_size"] = 2

data_params = {"x_min": 0., "x_max": 1.}

adversarial_args = dict(attack=PGD,
                        attack_args=dict(data_params=data_params,
                                         attack_params=attack_params),
                        verbose=False,
                        progress_bar=True)


# save_image(model, test_loader, attack_params,
#            attack_args=attack_args, attack_func=attack_func)

start_time = time.time()
black_box_test_args = dict(model=model,
                           substitute_model=substitute_model,
                           test_loader=test_loader,
                           adversarial_args=adversarial_args,
                           verbose=True,
                           progress_bar=True)

attack_loss, attack_acc = substitute_test(**black_box_test_args)

print(f'BB Attack  \t loss: {attack_loss:.4f} \t acc: {attack_acc:.4f}')
print(f"{time.time()-start_time:.2f} seconds")

adversarial_test_args = dict(model=model,
                             test_loader=test_loader,
                             adversarial_args=adversarial_args,
                             verbose=True,
                             progress_bar=True)

attack_loss, attack_acc = whitebox_test(**adversarial_test_args)

print(f'WB Attack  \t loss: {attack_loss:.4f} \t acc: {attack_acc:.4f}')
print(f"{time.time()-start_time:.2f} seconds")
