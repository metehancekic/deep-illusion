"""Adversarial attack module implemented for PyTorch"""


from ._fgsm import FGSM, FGSM_targeted, FGM
from ._rfgsm import RFGSM
from ._pgd import PGD, PGD_EOT, PGD_EOT_normalized, PGD_EOT_sign
from ._bim import BIM, BIM_EOT
from ._soft_attacks import soft_attack_single_step, iterative_soft_attack
from .._version import __version__

__all__ = ["FGSM", "FGSM_targeted", "FGM", "RFGSM", "PGD", "PGD_EOT", "PGD_EOT_normalized",
                   "PGD_EOT_sign", "BIM", "BIM_EOT", "soft_attack_single_step",
                   "iterative_soft_attack", "__version__"]
