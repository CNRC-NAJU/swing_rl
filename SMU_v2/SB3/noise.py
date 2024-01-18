"""
Action noise implementation of SB3
- All unneccessary codes of SMUD project are removed
- #! comments are from HY
"""

from abc import ABC, abstractmethod

import numpy as np


class ActionNoise(ABC):
    """
    The action noise base class
    #! Not sure what this is for or where it is used
    """

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        """
        Call end of episode reset for the noise
        """
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()
