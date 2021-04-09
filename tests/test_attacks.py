"""
python test_attacks.py --dataset=mnist --attack_method=PGD_EOT_normalized
"""


import time
import argparse

from deepillusion.torchattacks import FGSM, FGSM_targeted, RFGSM, PGD, PGD_EOT, PGD_EOT_normalized, PGD_EOT_sign, BIM, BIM_EOT, soft_attack_single_step, iterative_soft_attack, SPSA
from deepillusion.torchattacks.analysis import whitebox_test, substitute_test, get_perturbation_stats

from test_utils import initiate_cifar10, initiate_mnist


parser = argparse.ArgumentParser(description='Test module')


# Dataset
parser.add_argument('--dataset', type=str, default='mnist', choices=[
                    "mnist", "fashion", "cifar"], metavar='mnist/fashion/cifar', help='Which dataset to use (default: mnist)')

parser.add_argument('--attack_method', type=str, default='PGD', choices=[
                    "FGSM", "FGSM_targeted", "RFGSM", "BIM", "BIM_EOT", "PGD", "PGD_EOT", "PGD_EOT_normalized", "PGD_EOT_sign", "SPSA"], metavar='', help='Attack method')

parser.add_argument('--loss_function', type=str, default='cross_entropy', choices=[
                    "cross_entropy", "carlini_wagner"], metavar='', help='Loss function to be used for attack')

args = parser.parse_args()

if args.dataset == "cifar":
    model, model_2, train_loader, test_loader = initiate_cifar10()
    attack_params = {
        "norm": "inf",
        "eps": 8./255,
        "alpha": 10./255,
        "step_size": 2./255,
        "num_steps": 20,
        "num_samples": 20,
        "random_start": False,
        "num_restarts": 1,
        "EOT_size": 20,
        }
elif args.dataset == "mnist":
    model, model_2, train_loader, test_loader = initiate_mnist(args.dataset)
    attack_params = {
        "norm": "inf",
        "eps": 0.3,
        "alpha": 0.4,
        "step_size": 0.05,
        "num_steps": 100,
        "num_samples": 100,
        "random_start": False,
        "num_restarts": 10,
        "EOT_size": 20,
        }

elif args.dataset == "fashion":
    model, model_2, train_loader, test_loader = initiate_mnist(args.dataset)
    attack_params = {
        "norm": "inf",
        "eps": 0.1,
        "alpha": 0.125,
        "step_size": 0.004,
        "num_steps": 100,
        "num_samples": 10,
        "random_start": False,
        "num_restarts": 1,
        "EOT_size": 20,
        }

data_params = {"x_min": 0., "x_max": 1.}

attacks = dict(Standard=None,
               SPSA=SPSA,
               FGSM=FGSM,
               FGSM_targeted=FGSM_targeted,
               RFGSM=RFGSM,
               BIM=BIM,
               BIM_EOT=BIM_EOT,
               PGD=PGD,
               PGD_EOT=PGD_EOT,
               PGD_EOT_normalized=PGD_EOT_normalized,
               PGD_EOT_sign=PGD_EOT_sign,
               soft_attack_single_step=soft_attack_single_step,
               iterative_soft_attack=iterative_soft_attack)

attack_args = dict(data_params=data_params,
                   attack_params=attack_params,
                   loss_function=args.loss_function,
                   verbose=False)

if args.attack_method not in ["FGSM", "FGSM_targeted", "RFGSM"]:
    attack_args["progress_bar"] = True

adversarial_args = dict(attack=attacks[args.attack_method],
                        attack_args=attack_args)
# breakpoint()

start_time = time.time()

# substitute_test_args = dict(model=model,
#                             substitute_model=model_2,
#                             test_loader=test_loader,
#                             adversarial_args=adversarial_args,
#                             verbose=True,
#                             progress_bar=True)

# attack_loss, attack_acc = substitute_test(**substitute_test_args)

whitebox_test_args = dict(model=model_2,
                          test_loader=test_loader,
                          adversarial_args=adversarial_args,
                          verbose=True,
                          progress_bar=True)

attack_loss, attack_acc = whitebox_test(**whitebox_test_args)

print(f"{time.time()-start_time:.2f} seconds\n")
