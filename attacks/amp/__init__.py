"""
Accelerated attack codes using amp from apex module
"""

from .fgsm import FGSM
from .rfgsm import RFGSM
from .pgd import PGD
from .._version import __version__

from warnings import warn
warn("Using amp versions can cause gradient masking issue for overconfident models (i.e models for MNIST dataset)")

__all__ = ['FGSM', 'RFGSM', 'PGD', '__version__']
