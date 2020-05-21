"""Adversarial attack module implemented for PyTorch"""

from ._fgsm import FGSM, FGSM_targeted, FGM
from ._rfgsm import RFGSM
from ._pgd import PGD, ePGD, PEGD
from ._cw import CWlinf, CWlinf_e
from ._bim import BIM
from ._soft_attacks import soft_attack_single_step, iterative_soft_attack
from .._version import __version__

__all__ = ['FGSM', 'FGSM_targeted', "FGM", 'RFGSM', 'PGD', "CWlinf", "CWlinf_e", 'ePGD',
           'PEGD', 'BIM', 'soft_attack_single_step', 'iterative_soft_attack', '__version__']
