from functools import partial
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt
from config.grid import SwingConfig
from scipy.sparse import coo_matrix

from . import acceleration
from .rk import rk1, rk2, rk4

arr = npt.NDArray[np.float64]


class SwingSolver(Protocol):
    """For given phase, dphase, return the next phase and dphase after dt"""

    def __call__(self, phase: arr, dphase: arr) -> tuple[arr, arr]:
        ...


def get_swing_solver(
    config: SwingConfig,
    weighted_adjacency_matrix: arr,
    params: arr,
) -> SwingSolver:
    """
    Create SwingSolver with fixed adjacency matrix and parameters

    Args
    solver_name: which Runge-Kutta algorithm will be used
    weighted_adjacency_matrix: [N, N], adjacency matrix weighted by coupling constants
    params: [3, N] power, gamma, mass of each node
    """
    swing_acceleration: Callable[[arr, arr], arr]
    if "sparse" in config.name:
        swing_acceleration = partial(
            acceleration.swing_acceleration_sparse,
            coo_matrix(weighted_adjacency_matrix),
            params,
        )
    elif "original" in config.name:
        swing_acceleration = partial(
            acceleration.swing_acceleration_original,
            weighted_adjacency_matrix,
            params,
        )
    else:
        swing_acceleration = partial(
            acceleration.swing_acceleration_default,
            weighted_adjacency_matrix,
            params,
        )

    # Which order of Runge-Kutta to use
    if "rk1" in config.name:
        rk = rk1
    elif "rk2" in config.name:
        rk = rk2
    elif "rk4" in config.name:
        rk = rk4
    else:
        raise TypeError(f"No such solver name: {config.name}")

    return partial(rk, swing_acceleration=swing_acceleration, dt=config.dt)
