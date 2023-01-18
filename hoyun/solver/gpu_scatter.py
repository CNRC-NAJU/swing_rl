"""
Solve swing equation equation

m_i * d^2 theta_i / dt^2 = P_i - gamma_i * d theta_i/dt + sum_j K_ij * A_ij * sin(theta_j-theta_i)

Use pytorch to use gpu computation
Do GNN-like computation using torch_scatter
"""

import functools

import torch
from torch_scatter import scatter_sum


def get_acceleration(
    edge_list: torch.LongTensor,
    weights: torch.Tensor,
    params: torch.Tensor,
    phase: torch.Tensor,
    dphase: torch.Tensor,
) -> torch.Tensor:
    """
    edge_list: (2, E), same as torch_geometric convention
    weights: (E)
    phase: (N, )
    dphase: (N, )
    params: (3, ) power, gamma, mass
    """
    row, col = edge_list
    force = (
        params[0]
        - params[1] * dphase
        + scatter_sum(weights * (phase[row] - phase[col]).sin(), col)
    )
    return force / params[2]


def rk1(
    edge_list: torch.LongTensor,
    weights: torch.Tensor,
    params: torch.Tensor,
    phase: torch.Tensor,
    dphase: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return next phase, next dphase"""
    acceleration = get_acceleration(edge_list, weights, params, phase, dphase)
    velocity = dphase
    return phase + dt * velocity, dphase + dt * acceleration


def rk2(
    edge_list: torch.LongTensor,
    weights: torch.Tensor,
    params: torch.Tensor,
    phase: torch.Tensor,
    dphase: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return next phase, next dphase"""
    get_acc = functools.partial(get_acceleration, edge_list, weights, params)

    acceleration1 = get_acc(phase, dphase)
    velocity1 = dphase

    temp_phase = phase + dt * velocity1
    temp_dphase = dphase + dt * acceleration1
    acceleration2 = get_acc(temp_phase, temp_dphase)
    velocity2 = temp_dphase

    velocity = 0.5 * (velocity1 + velocity2)
    acceleration = 0.5 * (acceleration1 + acceleration2)
    return phase + dt * velocity, dphase + dt * acceleration


def rk4(
    edge_list: torch.LongTensor,
    weights: torch.Tensor,
    params: torch.Tensor,
    phase: torch.Tensor,
    dphase: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return next phase, next dphase"""
    get_acc = functools.partial(get_acceleration, edge_list, weights, params)

    acceleration1 = get_acc(phase, dphase)
    velocity1 = dphase

    temp_phase = phase + 0.5 * dt * velocity1
    temp_dphase = dphase + 0.5 * dt * acceleration1
    acceleration2 = get_acc(temp_phase, temp_dphase)
    velocity2 = temp_dphase

    temp_phase = phase + 0.5 * dt * velocity2
    temp_dphase = dphase + 0.5 * dt * acceleration2
    acceleration3 = get_acc(temp_phase, temp_dphase)
    velocity3 = temp_dphase

    temp_phase = phase + dt * velocity3
    temp_dphase = dphase + dt * acceleration3
    acceleration4 = get_acc(temp_phase, temp_dphase)
    velocity4 = temp_dphase

    velocity = (velocity1 + 2.0 * velocity2 + 2.0 * velocity3 + velocity4) / 6.0
    acceleartion = (
        acceleration1 + 2.0 * acceleration2 + 2.0 * acceleration3 + acceleration4
    ) / 6.0
    return phase + dt * velocity, dphase + dt * acceleartion
