import time

from deepillusion.torchattacks import FGSM, FGSM_targeted, RFGSM, PGD, BIM, soft_attack_single_step, iterative_soft_attack

from test_utils import initiate_cifar10, test_adversarial

model, train_loader, test_loader = initiate_cifar10()


data_params = {"x_min": 0., "x_max": 1.}
attack_params = {
    "norm": "inf",
    "eps": 8./255,
    "alpha": 10./255,
    "step_size": 2./255,
    "num_steps": 7,
    "random_start": False,
    "num_restarts": 1,
    }

attack_args = dict(net=model,
                   data_params=data_params,
                   attack_params=attack_params)
attack_func = BIM

start_time = time.time()
attack_loss, attack_acc = test_adversarial(model, test_loader, attack_params,
                                           attack_args=attack_args, attack_func=attack_func)
print(f'Attack  \t loss: {attack_loss:.4f} \t acc: {attack_acc:.4f}')
print(f"{time.time()-start_time:.2f} seconds")
