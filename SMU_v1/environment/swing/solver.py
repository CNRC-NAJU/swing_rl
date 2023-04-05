import functools
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt
from config import SwingConfig
from scipy.sparse import coo_matrix

from . import default, original, sparse


class Solver(Protocol):
    def __call__(
        self,
        phase: npt.NDArray[SwingConfig.dtype],
        dphase: npt.NDArray[SwingConfig.dtype],
    ) -> tuple[npt.NDArray[SwingConfig.dtype], npt.NDArray[SwingConfig.dtype]]:
        ...


def swing_solver(
    weighted_adjacency_matrix: npt.NDArray[SwingConfig.dtype] | coo_matrix,
    params: npt.NDArray[SwingConfig.dtype],
) -> Solver:
    if "sparse" in SwingConfig.name:
        solver_module = sparse
        weighted_adjacency_matrix = coo_matrix(weighted_adjacency_matrix)
    elif "original" in SwingConfig.name:
        solver_module = original
    else:
        solver_module = default

    # Which order of Runge-Kutta to use
    if "rk1" in SwingConfig.name:
        step_solver = solver_module.rk1
    elif "rk2" in SwingConfig.name:
        step_solver = solver_module.rk2
    elif "rk4" in SwingConfig.name:
        step_solver = solver_module.rk4
    else:
        raise ValueError(f"No such solver name: {SwingConfig.name}")

    return functools.partial(
        step_solver,
        weighted_adjacency_matrix=weighted_adjacency_matrix,
        params=params,
        dt=SwingConfig.dt,
    )
