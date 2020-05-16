"""Adversarial defense module implemented on Pytorch"""

from ._adversarial_train_test import adversarial_epoch, adversarial_test
from ..._version import __version__

__all__ = ['adversarial_epoch', 'adversarial_test', '__version__']
