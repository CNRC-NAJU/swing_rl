from typing import Protocol

import numpy as np
import numpy.typing as npt
from config.grid import SwingConfig

arr = npt.NDArray[np.float64]


class Monitor(Protocol):
    """Monitor the trajectory of swing equation and return True if early stop"""

    def __call__(self, phase: arr, dphase: arr, dt: float) -> bool:
        ...


class NullMonitor:
    """Always return False, no early stop"""

    def __call__(self, phase: arr, dphase: arr, dt: float) -> bool:
        return False


class InsideMonitor:
    """
    Monitor the absolute value of dphase.
    Return True if all values are inside the threshold range.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, phase: arr, dphase: arr, dt: float) -> bool:
        return np.all(np.abs(dphase) < self.threshold).item()


class OutsideMonitor:
    """
    Monitor the absolute value of dphase
    Return True if even one value is outside the threshold range
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, phase: arr, dphase: arr, dt: float) -> bool:
        return np.any(np.abs(dphase) > self.threshold).item()


def get_monitor(config: SwingConfig) -> Monitor:
    match config.monitor_name:
        case "null":
            return NullMonitor()
        case "inside":
            return InsideMonitor(config.monitor_threshold)
        case "outside":
            return OutsideMonitor(config.monitor_threshold)
