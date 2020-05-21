"""
Accelerated attack codes using amp from apex module
"""

from ._fgsm import FGSM, FGSM_targeted, FGM
from ._rfgsm import RFGSM
from ._pgd import PGD, ePGD, PEGD
from ._cw import CWlinf, CWlinf_e
from ._bim import BIM
from ._soft_attacks import soft_attack_single_step, iterative_soft_attack
from ..._version import __version__
from ..._utils import GradientMaskingWarning

from warnings import warn
warn("Using amp versions can cause gradient masking issue for overconfident models (i.e models for MNIST dataset)", GradientMaskingWarning)

__all__ = ['FGSM', 'FGM', 'FGSM_targeted', 'RFGSM', 'PGD', 'ePGD', 'PEGD', "CWlinf",
           "CWlinf_e", 'BIM', 'soft_attack_single_step', 'iterative_soft_attack', '__version__']
