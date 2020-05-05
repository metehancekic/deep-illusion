![alt text][logo]

[logo]: https://github.com/metehancekic/deep-illusion/blob/master/figs/confused-ai.png

### Deep Illusion #

Deep Illusion is a toolbox for adversarial attacks in machine learning. Current version is only implemented for Pytorch models. DeepIllusion is a growing and developing module day by day. Module currently includes complete implementation of well-known attacks (PGD, FGSM, R-FGSM etc..). All attacks have an apex(amp) version which you can run your attacks fast and accurately. We strongly recommend that amp versions should only be used for adversarial training since it may have gradient masking issues after neural net gets confident about its decisions. 


## Module Structure #

```
deep-illusion
│   README.md
│
|───deepillusion
|   |   _utils.py               Utility functions
|   |
|   |───torchattacks
|   |   │   _fgsm.py                     Fast Gradient Sign Method
|   |   │   _rfgsm.py                    Random Start + Fast Gradient Sign Method
|   |   │   _pgd.py                      Projected Gradient Descent
|   |   │   _soft_attacks.py             Soft attack functions
|   |   │ 
|   |   |───amp
|   |   |   │   _fgsm.py                     Mixed Precision (Faster) - Fast Gradient Sign Method
|   |   |   │   _rfgsm.py                    Mixed Precision (Faster) - Random Start + Fast Gradient Sign Method
|   |   |   │   _pgd.py                      Mixed Precision (Faster) - Projected Gradient Descent
|   |   |   |   _soft_attacks.py             Mixed Precision (Faster) - Soft attack functions
|   |   |
|   |   └───analysis
|   |       │   _perturbation_statistics     Perturbations statistics functions
|   |
|   |───tfattacks
|   |   |
|   |
|   └───jaxattacks
|       |
|
└───tests
    |   fgsm_test.py
    |   fgsmt_test.py
    |   pgd_test.py
    |   bim_test.py
    |   rfgsm_test.py
    |   test_utils.py
```
## Dependencies #

> numpy                     1.16.4\
> tqdm                      4.31.1

**torchattacks**
> pytorch                   1.4.0\
> apex                      0.1  (optional)

**tfattacks**
> tensorflow                   

**jaxattacks**
> jax

## Installation #

```bash
pip install deepillusion
```

## Example Use #

```python
from deepillusion.torchattacks import PGD, FGSM, RFGSM

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
attack_params = {"norm": "inf",
                 "eps": 8./255}
fgsm_args = dict(net=model,
                 x=data,
                 y_true=target,
                 data_params=data_params,
                 attack_params=attack_params)
perturbs = FGSM(**fgsm_args)
data_adversarial = data + perturbs
```
## Update #

- pip install deepillusion --upgrade

## Current Version #
0.0.6
