"""
Calculate acceleration of swing equation

m_i * d^2 theta_i / dt^2 = P_i - gamma_i * d theta_i/dt + sum_j K_ij * A_ij * sin(theta_j-theta_i)
"""

from typing import overload

import numpy as np
import numpy.typing as npt
import torch
from numba import njit
from scipy.sparse import coo_matrix
from torch_scatter import scatter_sum

arr32 = npt.NDArray[np.float32]
arr64 = npt.NDArray[np.float64]
tensor = torch.Tensor


@overload
def swing_acceleration_default(
    weighted_adjacency_matrix: arr32, params: arr32, phase: arr32, dphase: arr32
) -> arr32:
    ...


@overload
def swing_acceleration_default(
    weighted_adjacency_matrix: arr64, params: arr64, phase: arr64, dphase: arr64
) -> arr64:
    ...


@njit(fastmath=True)
def swing_acceleration_default(weighted_adjacency_matrix, params, phase, dphase):
    """
    Calculate interaction term as follows:
    sum_j K_ij * A_ij * sin(theta_j-theta_i)
    = cos(theta_i) * [KA @ sin(theta)]_i - sin(theta_i) * [KA @ cos(theta)]_i

    Args
    weighted adjacency matrix: [N, N]
    params: [3, N] power, gamma, mass of each node
    phase: [N, ], phase of each node
    dphase: [N, ], dphase(velocity) of each node

    Return
    acceleration: [N, ]
    """
    # Interaction
    sin_phase, cos_phase = np.sin(phase), np.cos(phase)
    sin_phase_adj = weighted_adjacency_matrix @ sin_phase
    cos_phase_adj = weighted_adjacency_matrix @ cos_phase

    force = (
        params[0]  # P
        - params[1] * dphase  # -gamma * velocity
        + cos_phase * sin_phase_adj  # interaction1
        - sin_phase * cos_phase_adj  # interaction2
    )
    return force / params[2]  # a = F / m


@overload
def swing_acceleration_original(
    weighted_adjacency_matrix: arr32, params: arr32, phase: arr32, dphase: arr32
) -> arr32:
    ...


@overload
def swing_acceleration_original(
    weighted_adjacency_matrix: arr64, params: arr64, phase: arr64, dphase: arr64
) -> arr64:
    ...


@njit(fastmath=True)
def swing_acceleration_original(weighted_adjacency_matrix, params, phase, dphase):
    """
    Args
    weighted adjacency matrix: [N, N]
    params: [3, N] power, gamma, mass of each node
    phase: [N, ], phase of each node
    dphase: [N, ], dphase(velocity) of each node

    Return
    acceleration: [N, ]
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
def swing_acceleration_sparse(
    weighted_adjacency_matrix: coo_matrix, params: arr32, phase: arr32, dphase: arr32
) -> arr32:
    ...


@overload
def swing_acceleration_sparse(
    weighted_adjacency_matrix: coo_matrix, params: arr64, phase: arr64, dphase: arr64
) -> arr64:
    ...


def swing_acceleration_sparse(weighted_adjacency_matrix, params, phase, dphase):
    """
    Use sparse matrix representation

    Args
    weighted adjacency matrix: [N, N]
    params: [3, N] power, gamma, mass of each node
    phase: [N, ], phase of each node
    dphase: [N, ], dphase(velocity) of each node

    Return
    acceleration: [N, ]
    """

    # Interaction
    sin_phase, cos_phase = np.sin(phase), np.cos(phase)
    sin_phase_adj = weighted_adjacency_matrix @ sin_phase
    cos_phase_adj = weighted_adjacency_matrix @ cos_phase

    force = (
        params[0]  # P
        - params[1] * dphase  # -gamma * velocity
        + cos_phase * sin_phase_adj  # interaction1
        - sin_phase * cos_phase_adj  # interaction2
    )
    return force / params[2]  # a = F / m


def swing_acceleration_gpu(
    weighted_adjacency_matrix: tensor,
    params: tensor,
    phase: tensor,
    dphase: tensor,
) -> tensor:
    """
    Calculate interaction term as follows:
    sum_j K_ij * A_ij * sin(theta_j-theta_i)
    = cos(theta_i) * [KA @ sin(theta)]_i - sin(theta_i) * [KA @ cos(theta)]_i

    Use pytorch to use gpu computation
    Sparse representation of adjacency matrix is also valid

    Args
    edge_list: [2, 2E], same as torch_geometric convention, undirected
    weights: [2E, ], coupling constant, undirected
    params: [3, N] power, gamma, mass of each node
    phase: [N, ], phase of each node
    dphase: [N, ], dphase(velocity) of each node

    Return
    acceleration: [N, ]
    """
    # Interaction
    sin_phase, cos_phase = torch.sin(phase), torch.cos(phase)
    sin_phase_adj = weighted_adjacency_matrix @ sin_phase
    cos_phase_adj = weighted_adjacency_matrix @ cos_phase

    force = (
        params[0]  # P
        - params[1] * dphase  # -gamma * velocity
        + cos_phase * sin_phase_adj  # interaction1
        - sin_phase * cos_phase_adj  # interaction2
    )
    return force / params[2]  # a = F / m


def swing_accleration_scatter(
    edge_list: torch.LongTensor,
    weights: tensor,
    params: tensor,
    phase: tensor,
    dphase: tensor,
) -> tensor:
    """
    Do GNN-like computation using torch_scatter

    Args
    edge_list: [2, 2E], same as torch_geometric convention, undirected
    weights: [2E, ], coupling constant, undirected
    params: [3, N] power, gamma, mass of each node
    phase: [N, ], phase of each node
    dphase: [N, ], dphase(velocity) of each node

    Return
    acceleration: [N, ]
    """
    row, col = edge_list
    force = (
        params[0]
        - params[1] * dphase
        + scatter_sum(weights * (phase[row] - phase[col]).sin(), col)
    )
    return force / params[2]
