from typing import TypeVar

import numpy as np
import numpy.typing as npt
from config import RL_CONFIG, SWING_CONFIG

T = TypeVar("T", np.float16, np.float32, np.float64, np.float128)
DTYPE = SWING_CONFIG.dtype


def normalize_phase(phase: npt.NDArray[T]) -> npt.NDArray[T]:
    """Normalize phase into (-pi, pi]"""
    return ((phase + np.pi) % (2 * np.pi) - np.pi).view(phase.dtype)


def is_stable(dphase: npt.NDArray[DTYPE]) -> bool:
    """Check if all dphase are smaller than eps
    For float32 precision, 1e-4 is adequte"""
    return np.all(np.abs(dphase) < RL_CONFIG.stable_threshold).item()


def is_failed(dphase: npt.NDArray[DTYPE]) -> int:
    """Return number of failed nodes"""
    return (dphase > RL_CONFIG.fail_threshold).sum()
