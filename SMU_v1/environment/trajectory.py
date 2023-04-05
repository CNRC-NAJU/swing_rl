from typing import TypeVar

import numpy as np
import numpy.typing as npt
from config import RLConfig

T = TypeVar("T", np.float16, np.float32, np.float64, np.float128)


def normalize_phase(phase: npt.NDArray[T]) -> npt.NDArray[T]:
    """Normalize phase into (-pi, pi]"""
    return ((phase + np.pi) % (2 * np.pi) - np.pi).view(phase.dtype)


def is_stable(dphase: npt.NDArray[T]) -> bool:
    """Check if all dphase are smaller than eps
    For float32 precision, 1e-4 is adequte"""
    return np.all(np.abs(dphase) < RLConfig().stable_threshold).item()


def is_failed(dphase: npt.NDArray[T]) -> int:
    """Return number of failed nodes"""
    return (dphase > RLConfig().fail_threshold).sum()
