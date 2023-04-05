import functools
from typing import Protocol

import numpy.typing as npt
from config import SwingConfig
from scipy.sparse import coo_matrix

from . import default, original, sparse

DTYPE = SwingConfig().dtype

class Solver(Protocol):
    def __call__(
        self,
        phase: npt.NDArray[DTYPE],
        dphase: npt.NDArray[DTYPE],
    ) -> tuple[npt.NDArray[DTYPE], npt.NDArray[DTYPE]]:
        ...


def swing_solver(
    weighted_adjacency_matrix: npt.NDArray[DTYPE] | coo_matrix,
    params: npt.NDArray[DTYPE],
) -> Solver:
    config = SwingConfig()
    if "sparse" in config.name:
        solver_module = sparse
        weighted_adjacency_matrix = coo_matrix(weighted_adjacency_matrix)
    elif "original" in config.name:
        solver_module = original
    else:
        solver_module = default

    # Which order of Runge-Kutta to use
    if "rk1" in config.name:
        step_solver = solver_module.rk1
    elif "rk2" in config.name:
        step_solver = solver_module.rk2
    elif "rk4" in config.name:
        step_solver = solver_module.rk4
    else:
        raise ValueError(f"No such solver name: {config.name}")

    return functools.partial(
        step_solver,
        weighted_adjacency_matrix=weighted_adjacency_matrix,
        params=params,
        dt=config.dt,
    )
