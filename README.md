![alt text][logo]

[logo]: https://github.com/metehancekic/deep-illusion/blob/master/figs/confused-ai.png

### Deep Illusion #

Deep Illusion is a toolbox for adversarial attacks in machine learning. Current version is only implemented for Pytorch models. DeepIllusion is a growing and developing python module which aims to help adversarial machine learning community to accelerate their research. Module currently includes complete implementation of well-known attacks (PGD, FGSM, R-FGSM, CW, BIM etc..). All attacks have an apex(amp) version which you can run your attacks fast and accurately. We strongly recommend that amp versions should only be used for adversarial training since it may have gradient masking issues after neural net gets confident about its decisions. All attack methods have an option (Verbose: False) to check if gradient masking is happening. 

All attack codes are written in functional programming style, therefore, users can easily call the method function and feed the input data and model to get perturbations. All codes are documented, and contains the example use in their description. Users can easily access the documentation by typing "??" at the and of the method they want to use in Ipython (E.g FGSM?? or PGD??). Output perturbations are already clipped for each image to prevent illegal pixel values. We are open to contributers to expand the attack methods arsenal.

We also include the most effective current approach to defend DNNs against adversarial perturbations which is training the network using adversarially perturbed examples. Adversarial training and testing methods are included in torchdefenses submodule. 

To standardize the arguments for all attacks, methods accept attack parameters as a dictionary named as attack_params which contains the necessary parameters for each attack. Furthermore, attack methods get the data properties such as the maximum and the minimum pixel value as another dictionary named data_params. These dictinaries make function calls concise and standard for all methods.

Current version is tested with different defense methods and the standard models for verification and we observed the reported accuracies.

Maintainers:
    [WCSL Lab](https://wcsl.ece.ucsb.edu), 
    [Metehan Cekic](https://www.ece.ucsb.edu/~metehancekic/), 
    [Can Bakiskan](https://wcsl.ece.ucsb.edu/people/can-bakiskan), 
    [Soorya Gopal](https://wcsl.ece.ucsb.edu/people/soorya-gopalakrishnan)


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
                verbose=False,
                progress_bar=False)               
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
0.1.9

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
|   |   │   _cw.py                       Carlini Wagner Linf
|   |   │   _bim.py                      Basic Iterative Method
|   |   │   _soft_attacks.py             Soft attack functions
|   |   │ 
|   |   |───amp
|   |   |   │   _fgsm.py                     Mixed Precision (Faster) - Fast Gradient Sign Method
|   |   |   │   _rfgsm.py                    Mixed Precision (Faster) - Random Start + Fast Gradient Sign Method
|   |   |   │   _cw.py                       Mixed Precision (Faster) - Carlini Wagner Linf
|   |   |   │   _pgd.py                      Mixed Precision (Faster) - Projected Gradient Descent
|   |   |   |   _soft_attacks.py             Mixed Precision (Faster) - Soft attack functions
|   |   |
|   |   └───analysis
|   |       │   _perturbation_statistics     Perturbations statistics functions
|   |
|   |───torchdefenses
│   |   |   _adversarial_train_test.py       Adversarial Training - Adversarial Testing
|   |   │   
|   |   └───amp
|   |       │   _adversarial_train_test.py     Mixed Precision (Faster) - Adversarial Training - Adversarial Testing 
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
    |   cw_test.py
    |   test_utils.py

```
