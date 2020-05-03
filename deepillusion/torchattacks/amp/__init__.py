"""
Accelerated attack codes using amp from apex module
"""

from ._fgsm import FGSM
from ._rfgsm import RFGSM
from ._pgd import PGD
from ._soft_attacks import soft_attack_single_step, iterative_soft_attack
from ..._version import __version__
from ..._utils import GradientMaskingWarning

from warnings import warn
warn("Using amp versions can cause gradient masking issue for overconfident models (i.e models for MNIST dataset)", GradientMaskingWarning)

__all__ = ['FGSM', 'RFGSM', 'PGD', 'soft_attack_single_step',
           'iterative_soft_attack', '__version__']
