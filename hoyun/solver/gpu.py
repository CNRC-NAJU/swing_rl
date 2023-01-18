"""
Solve swing equation equation

m_i * d^2 theta_i / dt^2 = P_i - gamma_i * d theta_i/dt + sum_j K_ij * A_ij * sin(theta_j-theta_i)

Calculate interaction term as follows:
sum_j K_ij * A_ij * sin(theta_j-theta_i)
= cos(theta_i) * [KA @ sin(theta)]_i - sin(theta_i) * [KA @ cos(theta)]_i

Use pytorch to use gpu computation
Sparse representation of adjacency matrix is also valid
"""

import functools

import torch


def get_acceleration(
    weighted_adjacency_matrix: torch.Tensor,
    params: torch.Tensor,
    phase: torch.Tensor,
    dphase: torch.Tensor,
) -> torch.Tensor:
    """
    weighted adjacency matrix: (N, N)
    phase: (N, )
    dphase: (N, )
    params: (3, ) power, gamma, mass
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


def rk1(
    weighted_adjacency_matrix: torch.Tensor,
    params: torch.Tensor,
    phase: torch.Tensor,
    dphase: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return next phase, next dphase"""
    acceleration = get_acceleration(weighted_adjacency_matrix, params, phase, dphase)
    velocity = dphase
    return phase + dt * velocity, dphase + dt * acceleration


def rk2(
    weighted_adjacency_matrix: torch.Tensor,
    params: torch.Tensor,
    phase: torch.Tensor,
    dphase: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return next phase, next dphase"""
    get_acc = functools.partial(get_acceleration, weighted_adjacency_matrix, params)

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
    weighted_adjacency_matrix: torch.Tensor,
    params: torch.Tensor,
    phase: torch.Tensor,
    dphase: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return next phase, next dphase"""
    get_acc = functools.partial(get_acceleration, weighted_adjacency_matrix, params)

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
