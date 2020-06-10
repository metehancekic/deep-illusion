"""
python test_attacks.py --dataset=mnist --attack_method=PGD_EOT_normalized
"""


import time
import argparse
from tqdm import tqdm


from deepillusion.torchattacks.analysis.plot import loss_landscape

from test_utils import initiate_cifar10, initiate_mnist


parser = argparse.ArgumentParser(description='Test module')


# Dataset
parser.add_argument('--dataset', type=str, default='cifar', choices=[
                    "mnist", "fashion", "cifar"], metavar='mnist/fashion/cifar', help='Which dataset to use (default: mnist)')

# parser.add_argument('--attack_method', type=str, default='PGD', choices=[
#                     "FGSM", "FGSM_targeted", "RFGSM", "BIM", "BIM_EOT", "PGD", "PGD_EOT", "PGD_EOT_normalized", "PGD_EOT_sign"], metavar='', help='Attack method')

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
        "num_steps": 7,
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
        "step_size": 0.01,
        "num_steps": 100,
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
        "random_start": False,
        "num_restarts": 1,
        "EOT_size": 20,
        }

data_params = {"x_min": 0., "x_max": 1.}


for i in tqdm(range(10)):
    loss_landscape_args = dict(model=model,
                               data_loader=test_loader,
                               second_direction="adversarial",
                               img_index=i,
                               fig_name="loss_landscape_" + args.dataset + "_i_" + str(i) + ".eps",
                               norm="inf",
                               z_axis_type="loss")

    loss_landscape(**loss_landscape_args)
