![alt text][logo]

[logo]: https://github.com/metehancekic/deep-illusion/blob/master/figs/confused-ai.png

# Adversarial Machine Learning

With the advent of more powerful parallel computation units and huge data, we are able to train much more complex and expressive deep neural networks. That is said, deep neural nets (DNN) found its use in a wide variety of fields, ranging from computer vision to game playing agents. They are performing better on some tasks than even human experts in those fields. Despite their incredible success, it is by now well known that they are susceptible to small and carefully designed perturbations which are imperceptible to humans. The fact that DNN's can easily be fooled is a great problem since they are also used in security critical applications such as self-driving cars. Recently, research community has put a great effort to robustify neural networks against these adversarial examples. Despite great attention of research community, there is not a powerful defense mechanism found, and it is shown that defending against adversarial examples are not an easy goal. 

As another group working on this field, we share our attack codes as a library. This library is a side product of our research, and since we use this in our research as well, we made sure it works correctly and as mentioned in the original papers. To sum up, deepillusion contains easy to use and properly implemented adversarial methods.

We are open to suggestions "metehancekic@ucsb.edu".

# Deep Illusion #

Deep Illusion is a toolbox for adversarial attacks in machine learning. Current version is only implemented for Pytorch models. DeepIllusion is a growing and developing python module which aims to help adversarial machine learning community to accelerate their research. Module currently includes complete implementation of well-known attacks (PGD, FGSM, R-FGSM, CW, BIM etc..). All attacks have an apex(amp) version which you can run your attacks fast and accurately. We strongly recommend that amp versions should only be used for adversarial training since it may have gradient masking issues after neural net gets confident about its decisions. All attack methods have an option (Verbose: False) to check if gradient masking is happening. 

All attack codes are written in functional programming style, therefore, users can easily call the method function and feed the input data and model to get perturbations. All codes are documented, and contains the example use in their description. Users can easily access the documentation by typing "??" at the and of the method they want to use in Ipython (E.g FGSM?? or PGD??). Output perturbations are already clipped for each image to prevent illegal pixel values. We are open to contributers to expand the attack methods arsenal.

We also include the most effective current approach to defend DNNs against adversarial perturbations which is training the network using adversarially perturbed examples. Adversarial training and testing methods are included in torchdefenses submodule. 

Current version is tested with different defense methods and the standard models for verification and we observed the reported accuracies.

Maintainers:
    [WCSL Lab](https://wcsl.ece.ucsb.edu), 
    [Metehan Cekic](https://www.ece.ucsb.edu/~metehancekic/), 
    [Can Bakiskan](https://wcsl.ece.ucsb.edu/people/can-bakiskan), 
    [Soorya Gopal](https://wcsl.ece.ucsb.edu/people/soorya-gopalakrishnan),
    [Ahmet Dundar Sezer](https://wcsl.ece.ucsb.edu/people/ahmet-sezer) 


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

The most recent stable version can be installed via python package installer "pip", or you can clone it from the git page.

```bash
pip install deepillusion
```
or 
```bash
git clone git@github.com:metehancekic/deep-illusion.git
```

## Example Use #

As mentioned earlier, our adversarial methods are functional instead of modular type. Therefore, all you need to get the perturbations is feeding input data and its labels along with the attack parameters. 

To standardize the arguments for all attacks, methods accept attack parameters as a dictionary named as attack_params which contains the necessary parameters for each attack. Furthermore, attack methods get the data properties such as the maximum and the minimum pixel value as another dictionary named data_params. These dictinaries make function calls concise and standard for all methods.

Following code snippets show PGD and FGSM usage.

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

Deepillusion is a growing and developing library, therefore we strongly recommend to upgrade deepillusion regularly:

```bash
pip install deepillusion --upgrade
```

## Current Version #


0.2.5

## Module Structure #

In case investigation of the source codes are needed, this is how our module is structured:

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
|   |   |   │   _rfgsm.py                    MP - Random Start + Fast Gradient Sign Method
|   |   |   │   _cw.py                       MP - Carlini Wagner Linf
|   |   |   │   _pgd.py                      MP - Projected Gradient Descent
|   |   |   |   _soft_attacks.py             MP - Soft attack functions
|   |   |
|   |   └───analysis
|   |       │   _perturbation_statistics     Perturbations statistics functions
|   |
|   |───torchdefenses
│   |   |   _adversarial_train_test.py       Adversarial Training - Adversarial Testing
|   |   │   
|   |   └───amp
|   |       │   _adversarial_train_test.py     MP (Faster) - Adversarial Training - Adversarial Testing 
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
## Sources #

- [PyPi page for the code](https://pypi.org/project/deepillusion/)

- [Git repo for the code](https://github.com/metehancekic/deep-illusion)

