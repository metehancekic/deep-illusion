"""
Auxilary tools
"""


__all__ = ["GradientMaskingError"]


class GradientMaskingError(ValueError):
    """Gradient masking error (false sense of robustness)"""

    def __init__(self, arg):
        super(GradientMaskingError, self).__init__()
        self.arg = arg


class GradientMaskingWarning(Warning):
    """Gradient masking warning (false sense of robustness)"""

    def __init__(self, msg):
        super(GradientMaskingWarning, self).__init__(msg)
