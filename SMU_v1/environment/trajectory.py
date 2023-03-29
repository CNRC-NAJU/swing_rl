import numpy as np
import numpy.typing as npt
from config import RLConfig

arr32 = npt.NDArray[np.float32]


def normalize_phase(phase: arr32) -> arr32:
    """Normalize phase into (-pi, pi]"""
    return (phase + np.pi) % (2 * np.pi) - np.pi


def is_stable(dphase: arr32) -> bool:
    """Check if all dphase are smaller than eps
    For float32 precision, 1e-4 is adequte"""
    return np.all(np.abs(dphase) < RLConfig.stable_threshold).item()


def is_failed(dphase: arr32) -> int:
    """Return number of failed nodes"""
    return (dphase > RLConfig.fail_threshold).sum()
