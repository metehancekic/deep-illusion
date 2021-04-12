"""Adversarial attack module implemented for PyTorch"""


from ._fgsm import FGSM, FGSM_targeted, FGM
from ._rfgsm import RFGSM, RFGM
from ._pgd import PGD, PGD_EOT, PGD_EOT_normalized, PGD_EOT_sign, PGD_smooth
from ._bim import BIM, BIM_EOT
from ._spsa import SPSA
from ._soft_attacks import soft_attack_single_step, iterative_soft_attack
from .._version import __version__

__all__ = ["FGSM", "FGSM_targeted", "FGM", "RFGSM", "RFGM", "PGD", "PGD_EOT", "PGD_smooth", "PGD_EOT_normalized",
                   "PGD_EOT_sign", "BIM", "BIM_EOT", "SPSA", "soft_attack_single_step",
                   "iterative_soft_attack", "__version__"]
