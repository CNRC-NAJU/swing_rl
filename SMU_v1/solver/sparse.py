"""
Solve swing equation equation

m_i * d^2 theta_i / dt^2 = P_i - gamma_i * d theta_i/dt + sum_j K_ij * A_ij * sin(theta_j-theta_i)

Calculate interaction term as follows:
sum_j K_ij * A_ij * sin(theta_j-theta_i)
= cos(theta_i) * [KA @ sin(theta)]_i - sin(theta_i) * [KA @ cos(theta)]_i

Use sparse matrix representation
"""

import functools
from typing import Callable, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix

T = TypeVar("T", np.float32, np.float64)


def get_acceleration(
    weighted_adjacency_matrix: coo_matrix,
    params: npt.NDArray[T],
    phase: npt.NDArray[T],
    dphase: npt.NDArray[T],
) -> npt.NDArray[T]:
    """
    Return acceleration

    weighted adjacency matrix: (N, N)
    params: (3, N) power, gamma, mass
    phase: (N, )
    dphase: (N, )
    """

    # Interaction
    sin_phase, cos_phase = np.sin(phase), np.cos(phase)
    sin_phase_adj = weighted_adjacency_matrix @ sin_phase
    cos_phase_adj = weighted_adjacency_matrix @ cos_phase

    force: npt.NDArray[T] = (
        params[0]  # P
        - params[1] * dphase  # -gamma * velocity
        + cos_phase * sin_phase_adj  # interaction1
        - sin_phase * cos_phase_adj  # interaction2
    )
    return force / params[2]  # a = F / m


def rk1(
    weighted_adjacency_matrix: coo_matrix,
    params: npt.NDArray[T],
    phase: npt.NDArray[T],
    dphase: npt.NDArray[T],
    dt: npt.NDArray[T],
) -> tuple[npt.NDArray[T], npt.NDArray[T]]:
    """
    Return next phase, next dphase

    weighted adjacency matrix: (N, N)
    params: (3, N) power, gamma, mass
    phase: (N, )
    dphase: (N, )
    dt: (1, )
    """
    dtype = phase.dtype

    acceleration = get_acceleration(weighted_adjacency_matrix, params, phase, dphase)
    velocity = dphase
    return (phase + dt * velocity).view(dtype), (dphase + dt * acceleration).view(dtype)


def rk2(
    weighted_adjacency_matrix: coo_matrix,
    params: npt.NDArray[T],
    phase: npt.NDArray[T],
    dphase: npt.NDArray[T],
    dt: npt.NDArray[T],
) -> tuple[npt.NDArray[T], npt.NDArray[T]]:
    """
    Return next phase, next dphase

    weighted adjacency matrix: (N, N)
    params: (3, N) power, gamma, mass
    phase: (N, )
    dphase: (N, )
    dt: (1, )
    """
    dtype = phase.dtype
    get_acc: Callable[
        [npt.NDArray[T], npt.NDArray[T]], npt.NDArray[T]
    ] = functools.partial(get_acceleration, weighted_adjacency_matrix, params)

    acceleration1 = get_acc(phase, dphase)
    velocity1 = dphase

    temp_phase = (phase + dt * velocity1).view(dtype)
    temp_dphase = (dphase + dt * acceleration1).view(dtype)
    acceleration2 = get_acc(temp_phase, temp_dphase)
    velocity2 = temp_dphase

    velocity = (0.5 * (velocity1 + velocity2)).view(dtype)
    acceleration = (0.5 * (acceleration1 + acceleration2)).view(dtype)
    return (phase + dt * velocity).view(dtype), (dphase + dt * acceleration).view(dtype)


def rk4(
    weighted_adjacency_matrix: coo_matrix,
    params: npt.NDArray[T],
    phase: npt.NDArray[T],
    dphase: npt.NDArray[T],
    dt: npt.NDArray[T],
) -> tuple[npt.NDArray[T], npt.NDArray[T]]:
    """
    Return next phase, next dphase

    weighted adjacency matrix: (N, N)
    params: (3, N) power, gamma, mass
    phase: (N, )
    dphase: (N, )
    dt: (1, )
    """
    dtype = phase.dtype
    get_acc: Callable[
        [npt.NDArray[T], npt.NDArray[T]], npt.NDArray[T]
    ] = functools.partial(get_acceleration, weighted_adjacency_matrix, params)

    acceleration1 = get_acc(phase, dphase)
    velocity1 = dphase

    temp_phase = (phase + 0.5 * dt * velocity1).view(dtype)
    temp_dphase = (dphase + 0.5 * dt * acceleration1).view(dtype)
    acceleration2 = get_acc(temp_phase, temp_dphase)
    velocity2 = temp_dphase

    temp_phase = (phase + 0.5 * dt * velocity2).view(dtype)
    temp_dphase = (dphase + 0.5 * dt * acceleration2).view(dtype)
    acceleration3 = get_acc(temp_phase, temp_dphase)
    velocity3 = temp_dphase

    temp_phase = (phase + dt * velocity3).view(dtype)
    temp_dphase = (dphase + dt * acceleration3).view(dtype)
    acceleration4 = get_acc(temp_phase, temp_dphase)
    velocity4 = temp_dphase

    velocity = ((velocity1 + 2.0 * velocity2 + 2.0 * velocity3 + velocity4) / 6.0).view(
        dtype
    )
    acceleartion = (
        (acceleration1 + 2.0 * acceleration2 + 2.0 * acceleration3 + acceleration4)
        / 6.0
    ).view(dtype)
    return (phase + dt * velocity).view(dtype), (dphase + dt * acceleartion).view(dtype)