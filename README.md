### Pytorch Adversarial Attacks #

Complete implementation of well-known attacks (PGD, FGSM, R-FGSM etc..). All attacks have an apex(amp) version which you can run your attacks fast and accurately. We strongly recommend that amp versions should only be used for adversarial training since it may have gradient masking issues after neural net gets confident about its decisions. 


## Module Structure #

```
pytorch-adversarial-attacks
│   README.md
│
└───attacks
    │   _fgsm.py                     Fast Gradient Sign Method
    │   _rfgsm.py                    Random Start + Fast Gradient Sign Method
    │   _pgd.py                      Projected Gradient Descent
    │   _soft_attacks.py             Soft attack functions
    │ 
    |───amp
    |   │   _fgsm.py                     Mixed Precision (Faster) - Fast Gradient Sign Method
    |   │   _rfgsm.py                    Mixed Precision (Faster) - Random Start + Fast Gradient Sign Method
    |   │   _pgd.py                      Mixed Precision (Faster) - Projected Gradient Descent
    |   |   _soft_attacks.py             Mixed Precision (Faster) - Soft attack functions
    |   
    └───analysis
        │   _perturbation_statistics     Perturbations statistics functions
```
## Requirements #

Required libraries to be able to use attacks module.

> numpy                     1.16.4\
> pytorch                   1.4.0\
> tqdm                      4.31.1\
> apex                      0.1  (optional)

## Example Use #

Firstly, make sure that you've added the folder path (i.e .../pythorch-adversarial-attacks/) that includes attacks folder into environment variable PYTHONPATH. You can add following code to ~/.bashrc to permanently add attacks as module or use it on terminal to add attacks to pythonpath temporarily.
```bash
git clone git@github.com:metehancekic/pytorch-adversarial-attacks.git
export PYTHONPATH="${PYTHONPATH}:.../path_to_attacks/"
```

Import the adversarial attack functions from attacks folder as following

```python
from attacks import PGD, FGSM, RFGSM

##### PGD ######
data_params = {"x_min": 0., "x_max": 1.}
attack_params = {
    "norm": "inf",
    "eps": 8./255,
    "step_size": 2./255,
    "num_steps": 7,
    "random_start": False,
    "num_restarts": 1}
    
pgd_args = dict(net=model,
                x=data,
                y_true=target,
                data_params=data_params,
                attack_params=attack_params,
                verbose=False)               
perturbs = PGD(**pgd_args)
data_adversarial = data + perturbs

##### FGSM #####
data_params = {"x_min": 0., "x_max": 1.}
fgsm_args = dict(net=model,
                 x=data,
                 y_true=target,
                 eps=8.0/255,
                 data_params=data_params
                 norm="inf")
perturbs = FGSM(**fgsm_args)
data_adversarial = data + perturbs
```
## Update #

To update it to the most recent version, get into the folder of pythorch-adversarial-attacks/ and pull the repository.

## Version #
0.0.1
