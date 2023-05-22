"""
Solve swing equation

m_i * d^2 theta_i / dt^2 = P_i - gamma_i * d theta_i/dt + sum_j K_ij * A_ij * sin(theta_j-theta_i)

naive implementation

compiled with jit
"""

from typing import overload

import numpy as np
import numpy.typing as npt
from numba import njit

arr32 = npt.NDArray[np.float32]
arr64 = npt.NDArray[np.float64]
arr = np.ndarray


@overload
def get_acceleration(
    weighted_adjacency_matrix: arr32, params: arr32, phase: arr32, dphase: arr32
) -> arr32:
    ...


@overload
def get_acceleration(
    weighted_adjacency_matrix: arr64, params: arr64, phase: arr64, dphase: arr64
) -> arr64:
    ...


@njit(fastmath=True)
def get_acceleration(
    weighted_adjacency_matrix: arr, params: arr, phase: arr, dphase: arr
) -> arr:
    """
    Return acceleration

    weighted adjacency matrix: (N, N)
    params: (3, N) power, gamma, mass
    phase: (N, )
    dphase: (N, )
    """
    num_nodes = len(weighted_adjacency_matrix)
    force = params[0] - params[1] * dphase  # P - gamma * velocity

    for node in range(num_nodes):
        for neighbor in range(node + 1, num_nodes):
            # Filter neighbor
            if weighted_adjacency_matrix[node][neighbor] == 0:
                continue

            # Calculate interaction: KA sin(theta_j - theta_i)
            interaction = weighted_adjacency_matrix[node][neighbor] * np.sin(
                phase[neighbor] - phase[node]
            )

            # Use symmetric property
            force[node] += interaction
            force[neighbor] -= interaction

    return force / params[2]  # a = F / m


@overload
def rk1(
    weighted_adjacency_matrix: arr32,
    params: arr32,
    phase: arr32,
    dphase: arr32,
    dt: arr32,
) -> tuple[arr32, arr32]:
    ...


@overload
def rk1(
    weighted_adjacency_matrix: arr64,
    params: arr64,
    phase: arr64,
    dphase: arr64,
    dt: arr64,
) -> tuple[arr64, arr64]:
    ...


@njit(fastmath=True)
def rk1(
    weighted_adjacency_matrix: arr, params: arr, phase: arr, dphase: arr, dt: arr
) -> tuple[arr, arr]:
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


@overload
def rk2(
    weighted_adjacency_matrix: arr32,
    params: arr32,
    phase: arr32,
    dphase: arr32,
    dt: arr32,
) -> tuple[arr32, arr32]:
    ...


@overload
def rk2(
    weighted_adjacency_matrix: arr64,
    params: arr64,
    phase: arr64,
    dphase: arr64,
    dt: arr64,
) -> tuple[arr64, arr64]:
    ...


@njit(fastmath=True)
def rk2(
    weighted_adjacency_matrix: arr, params: arr, phase: arr, dphase: arr, dt: arr
) -> tuple[arr, arr]:
    """
    Return next phase, next dphase

    weighted adjacency matrix: (N, N)
    params: (3, N) power, gamma, mass
    phase: (N, )
    dphase: (N, )
    dt: (1, )
    """
    dtype = phase.dtype
    half = np.array(0.5, dtype=dtype)

    acceleration1 = get_acceleration(weighted_adjacency_matrix, params, phase, dphase)
    velocity1 = dphase

    temp_phase = (phase + dt * velocity1).view(dtype)
    temp_dphase = (dphase + dt * acceleration1).view(dtype)
    acceleration2 = get_acceleration(
        weighted_adjacency_matrix, params, temp_phase, temp_dphase
    )
    velocity2 = temp_dphase

    velocity = (half * (velocity1 + velocity2)).view(dtype)
    acceleration = (half * (acceleration1 + acceleration2)).view(dtype)
    return (phase + dt * velocity).view(dtype), (dphase + dt * acceleration).view(dtype)


@overload
def rk4(
    weighted_adjacency_matrix: arr32,
    params: arr32,
    phase: arr32,
    dphase: arr32,
    dt: arr32,
) -> tuple[arr32, arr32]:
    ...


@overload
def rk4(
    weighted_adjacency_matrix: arr64,
    params: arr64,
    phase: arr64,
    dphase: arr64,
    dt: arr64,
) -> tuple[arr64, arr64]:
    ...


@njit(fastmath=True)
def rk4(
    weighted_adjacency_matrix: arr, params: arr, phase: arr, dphase: arr, dt: arr
) -> tuple[arr, arr]:
    """
    Return next phase, next dphase

    weighted adjacency matrix: (N, N)
    params: (3, N) power, gamma, mass
    phase: (N, )
    dphase: (N, )
    dt: (1, )
    """
    dtype = phase.dtype
    half = np.array(0.5, dtype=dtype)

    acceleration1 = get_acceleration(weighted_adjacency_matrix, params, phase, dphase)
    velocity1 = dphase

    temp_phase = (phase + half * dt * velocity1).view(dtype)
    temp_dphase = (dphase + half * dt * acceleration1).view(dtype)
    acceleration2 = get_acceleration(
        weighted_adjacency_matrix, params, temp_phase, temp_dphase
    )
    velocity2 = temp_dphase

    temp_phase = (phase + half * dt * velocity2).view(dtype)
    temp_dphase = (dphase + half * dt * acceleration2).view(dtype)
    acceleration3 = get_acceleration(
        weighted_adjacency_matrix, params, temp_phase, temp_dphase
    )
    velocity3 = temp_dphase

    temp_phase = (phase + dt * velocity3).view(dtype)
    temp_dphase = (dphase + dt * acceleration3).view(dtype)
    acceleration4 = get_acceleration(
        weighted_adjacency_matrix, params, temp_phase, temp_dphase
    )
    velocity4 = temp_dphase

    velocity = (
        (
            velocity1
            + np.array(2.0, dtype=dtype) * velocity2
            + np.array(2.0, dtype=dtype) * velocity3
            + velocity4
        )
        / np.array(6.0, dtype=dtype)
    ).view(dtype)
    acceleartion = (
        (
            acceleration1
            + np.array(2.0, dtype=dtype) * acceleration2
            + np.array(2.0, dtype=dtype) * acceleration3
            + acceleration4
        )
        / np.array(6.0, dtype=dtype)
    ).view(dtype)
    return (phase + dt * velocity).view(dtype), (dphase + dt * acceleartion).view(dtype)
