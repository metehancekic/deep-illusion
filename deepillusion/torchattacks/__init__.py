"""Adversarial attack module implemented for PyTorch"""

from ._fgsm import FGSM
from ._rfgsm import RFGSM
from ._pgd import PGD
from ._soft_attacks import soft_attack_single_step, iterative_soft_attack
from .._version import __version__

__all__ = ['FGSM', 'RFGSM', 'PGD', 'soft_attack_single_step',
           'iterative_soft_attack', '__version__']
