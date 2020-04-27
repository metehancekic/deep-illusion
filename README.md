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
