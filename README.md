### Pytorch Adversarial Attacks #

Complete implementation of well-known attacks (PGD, FGSM, R-FGSM etc..). All attacks have an apex(amp) version which you can run your attacks fast and accurately. These attack functions firstly aim for robust attack, in case of well-known phenomenon "gradient masking" gives an error. 


## Module Structure #

```
adversary
│   README.md
|   utils.py                        Utility functions
│
└───gradient_based_attacks
    │   norm_ball_attacks.py        FGSM, R-FGSM, PGD attack functions
    │   soft_attacks.py             Soft attack functions
    │ 
    └───amp
        │   norm_ball_attacks.py        Amp versions of FGSM, R-FGSM, PGD attack functions
        │   soft_attacks.py             Amp versions of  soft attack functions
```
