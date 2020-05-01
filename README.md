### Pytorch Adversarial Attacks #

Complete implementation of well-known attacks (PGD, FGSM, R-FGSM etc..). All attacks have an apex(amp) version which you can run your attacks fast and accurately. We strongly recommend that amp versions should only be used for adversarial training. 


## Module Structure #

```
pytorch-adversarial-attacks
│   README.md
│
└───attacks
    │   fgsm.py                     Fast Gradient Sign Method
    │   rfgsm.py                    Random Start + Fast Gradient Sign Method
    │   pgd.py                      Projected Gradient Descent
    │   soft_attacks.py             Soft attack functions
    |   utils.py                    Utility functions
    │ 
    └───amp
        │   fgsm.py                     Mixed Precision (Faster) - Fast Gradient Sign Method
        │   rfgsm.py                    Mixed Precision (Faster) - Random Start + Fast Gradient Sign Method
        │   pgd.py                      Mixed Precision (Faster) - Projected Gradient Descent
        │   soft_attacks.py             Mixed Precision (Faster) - Soft attack functions
```

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
    "num_restarts": 1,
    }
    
pgd_args = dict(net=model,
                x=data,
                y_true=target,
                data_params=data_params,
                attack_params=attack_params,
                verbose=False)
                
perturbs = PGD(**pgd_args)

##### FGSM #####

data_params = {"x_min": 0., "x_max": 1.}
fgsm_args = dict(net=model,
                 x=data,
                 y_true=target,
                 eps=8.0/255,
                 data_params=data_params
                 norm="inf")
FGSM(**fgsm_args)


```
