import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from numba import njit, prange
from torch_scatter import scatter_sum


@njit
def get_velocity(dphase: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return dphase


@njit  # same as @jit(nopython=True)
def get_acceleration(
    adj_matrix: npt.NDArray[np.int64],
    phase: npt.NDArray[np.float64],
    dphase: npt.NDArray[np.float64],
    power: npt.NDArray[np.float64],
    mass: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    K: float,
) -> npt.NDArray[np.float64]:
    num_nodes = len(phase)

    acceleration = [0.0 for _ in range(num_nodes)]

    # Interaction
    for node in range(num_nodes):
        # F = P - gamma*velocity
        acceleration[node] += power[node] - gamma[node] * dphase[node]
        for neighbor in range(node + 1, num_nodes):
            if adj_matrix[node][neighbor] == 0:
                continue
            interaction = K * np.sin(phase[neighbor] - phase[node])
            acceleration[node] += interaction
            acceleration[neighbor] -= interaction

    # a = F/m
    for node in range(num_nodes):
        acceleration[node] /= mass[node]
    return np.array(acceleration, dtype=np.float64)


@njit(parallel=True, fastmath=True)
def get_acceleration_parallel(
    adj_matrix: npt.NDArray[np.int64],
    phase: npt.NDArray[np.float64],
    dphase: npt.NDArray[np.float64],
    power: npt.NDArray[np.float64],
    mass: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    K: float,
) -> npt.NDArray[np.float64]:
    num_nodes = len(phase)

    # F = P - gamma*velocity
    acceleration = power - gamma * dphase

    # Interaction
    for node in prange(num_nodes):
        for neighbor in range(node + 1, num_nodes):
            if adj_matrix[node][neighbor] == 0:
                continue
            interaction = K * np.sin(phase[neighbor] - phase[node])
            acceleration[node] += interaction
            acceleration[neighbor] -= interaction

    # a = F/m
    return acceleration / mass


class AccelerationTorch(nn.Module):
    def __init__(
        self,
        edge_index: torch.LongTensor,
        power: torch.Tensor,
        mass: torch.Tensor,
        gamma: torch.Tensor,
        K: torch.Tensor,
    ) -> None:
        """
        edge_index: [2, E] with max entry N-1, where N is number of nodes, E is number of edges
                    If edge (0, 1) exists, (1, 0) should not be exist
        """
        super().__init__()
        self.num_nodes = len(power)
        self.num_edges = len(edge_index[0])

        # Store edge-related variables
        self.row, self.col = edge_index
        self.edge_index_undirected = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        self.K = K

        # Store node-related variables
        # self.power = power
        self.mass = mass
        self.gamma = gamma

    def forward(
        self,
        phase: torch.Tensor,
        dphase: torch.Tensor,
        power: torch.Tensor,
    ) -> torch.Tensor:
        # F = P - gamma*velocity
        acceleration = power - self.gamma * dphase

        # Calculate interaction term for each edges
        interaction = self.K * torch.sin(phase[self.col] - phase[self.row])

        # Aggregate interaction to node acceleration
        interaction_undirected = torch.cat([interaction, -interaction], dim=0)
        acceleration += scatter_sum(
            interaction_undirected,
            self.edge_index_undirected[0],
            dim=0,
            dim_size=self.num_nodes,
        )

        # a = F/m
        return acceleration / self.mass

    def to(self, device: torch.device) -> None:
        """
        All tensors to device
        """
        # edge-related variables
        self.row = self.row.to(device)
        self.col = self.col.to(device)
        self.edge_index_undirected = self.edge_index_undirected.to(device)
        self.K = self.K.to(device)

        # node-related variables
        # self.power = self.power.to(device)
        self.mass = self.mass.to(device)
        self.gamma = self.gamma.to(device)


@njit
def rk2(
    adj_matrix: npt.NDArray[np.int64],
    phase: npt.NDArray[np.float64],
    dphase: npt.NDArray[np.float64],
    power: npt.NDArray[np.float64],
    mass: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    K: float,
    dt: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    delta_dphase1 = get_acceleration(adj_matrix, phase, dphase, power, mass, gamma, K)
    delta_phase1 = get_velocity(dphase)
    phase_tmp = phase + dt * delta_phase1
    dphase_tmp = dphase + dt * delta_dphase1

    delta_dphase2 = get_acceleration(
        adj_matrix, phase_tmp, dphase_tmp, power, mass, gamma, K
    )
    delta_phase2 = get_velocity(dphase_tmp)

    return (delta_dphase1 + delta_dphase2) / 2.0, (delta_phase1 + delta_phase2) / 2.0


@njit(fastmath=True)
def rk2_parallel(
    adj_matrix: npt.NDArray[np.int64],
    phase: npt.NDArray[np.float64],
    dphase: npt.NDArray[np.float64],
    power: npt.NDArray[np.float64],
    mass: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    K: float,
    dt: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    delta_dphase1 = get_acceleration_parallel(
        adj_matrix, phase, dphase, power, mass, gamma, K
    )
    delta_phase1 = get_velocity(dphase)
    phase_tmp = phase + dt * delta_phase1
    dphase_tmp = dphase + dt * delta_dphase1

    delta_dphase2 = get_acceleration_parallel(
        adj_matrix, phase_tmp, dphase_tmp, power, mass, gamma, K
    )
    delta_phase2 = get_velocity(dphase_tmp)

    return (delta_dphase1 + delta_dphase2) / 2.0, (delta_phase1 + delta_phase2) / 2.0


@torch.no_grad()
def rk2_torch(
    acceleration_torch: AccelerationTorch,
    phase: torch.Tensor,
    dphase: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta_dphase1 = acceleration_torch(phase, dphase)
    delta_phase1 = dphase
    phase_tmp = phase + dt * delta_phase1
    dphase_tmp = dphase + dt * delta_dphase1

    delta_dphase2 = acceleration_torch(phase_tmp, dphase_tmp)
    delta_phase2 = dphase_tmp

    return (delta_dphase1 + delta_dphase2) / 2.0, (delta_phase1 + delta_phase2) / 2.0


@njit
def rk4(
    adj_matrix: npt.NDArray[np.int64],
    phase: npt.NDArray[np.float64],
    dphase: npt.NDArray[np.float64],
    power: npt.NDArray[np.float64],
    mass: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    K: float,
    dt: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    delta_dphase1 = get_acceleration(adj_matrix, phase, dphase, power, mass, gamma, K)
    delta_phase1 = get_velocity(dphase)
    phase_tmp = phase + 0.5 * dt * delta_phase1
    dphase_tmp = dphase + 0.5 * dt * delta_dphase1

    delta_dphase2 = get_acceleration(
        adj_matrix, phase_tmp, dphase_tmp, power, mass, gamma, K
    )
    delta_phase2 = get_velocity(dphase_tmp)
    phase_tmp = phase + 0.5 * dt * delta_phase2
    dphase_tmp = dphase + 0.5 * dt * delta_dphase2

    delta_dphase3 = get_acceleration(
        adj_matrix, phase_tmp, dphase_tmp, power, mass, gamma, K
    )
    delta_phase3 = get_velocity(dphase_tmp)
    phase_tmp = phase + dt * delta_phase3
    dphase_tmp = dphase + dt * delta_dphase3

    delta_dphase4 = get_acceleration(
        adj_matrix, phase_tmp, dphase_tmp, power, mass, gamma, K
    )
    delta_phase4 = get_velocity(dphase_tmp)

    return (
        (delta_dphase1 + 2 * delta_dphase2 + 2 * delta_dphase3 + delta_dphase4) / 6,
        (delta_phase1 + 2 * delta_phase2 + 2 * delta_phase3 + delta_phase4) / 6,
    )


@njit(fastmath=True)
def rk4_parallel(
    adj_matrix: npt.NDArray[np.int64],
    phase: npt.NDArray[np.float64],
    dphase: npt.NDArray[np.float64],
    power: npt.NDArray[np.float64],
    mass: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    K: float,
    dt: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    delta_dphase1 = get_acceleration_parallel(
        adj_matrix, phase, dphase, power, mass, gamma, K
    )
    delta_phase1 = get_velocity(dphase)
    phase_tmp = phase + 0.5 * dt * delta_phase1
    dphase_tmp = dphase + 0.5 * dt * delta_dphase1

    delta_dphase2 = get_acceleration_parallel(
        adj_matrix, phase_tmp, dphase_tmp, power, mass, gamma, K
    )
    delta_phase2 = get_velocity(dphase_tmp)
    phase_tmp = phase + 0.5 * dt * delta_phase2
    dphase_tmp = dphase + 0.5 * dt * delta_dphase2

    delta_dphase3 = get_acceleration_parallel(
        adj_matrix, phase_tmp, dphase_tmp, power, mass, gamma, K
    )
    delta_phase3 = get_velocity(dphase_tmp)
    phase_tmp = phase + dt * delta_phase3
    dphase_tmp = dphase + dt * delta_dphase3

    delta_dphase4 = get_acceleration_parallel(
        adj_matrix, phase_tmp, dphase_tmp, power, mass, gamma, K
    )
    delta_phase4 = get_velocity(dphase_tmp)

    return (
        (delta_dphase1 + 2 * delta_dphase2 + 2 * delta_dphase3 + delta_dphase4) / 6,
        (delta_phase1 + 2 * delta_phase2 + 2 * delta_phase3 + delta_phase4) / 6,
    )


@torch.no_grad()
def rk4_torch(
    acceleration_torch: AccelerationTorch,
    phase: torch.Tensor,
    dphase: torch.Tensor,
    power: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta_dphase1 = acceleration_torch(phase, dphase, power)
    delta_phase1 = dphase
    phase_tmp = phase + 0.5 * dt * delta_phase1
    dphase_tmp = dphase + 0.5 * dt * delta_dphase1

    delta_dphase2 = acceleration_torch(phase_tmp, dphase_tmp, power)
    delta_phase2 = dphase_tmp.clone()
    phase_tmp = phase + 0.5 * dt * delta_phase2
    dphase_tmp = dphase + 0.5 * dt * delta_dphase2

    delta_dphase3 = acceleration_torch(phase_tmp, dphase_tmp, power)
    delta_phase3 = dphase_tmp.clone()
    phase_tmp = phase + dt * delta_phase3
    dphase_tmp = dphase + dt * delta_dphase3

    delta_dphase4 = acceleration_torch(phase_tmp, dphase_tmp, power)
    delta_phase4 = dphase_tmp.clone()

    return (
        (delta_dphase1 + 2 * delta_dphase2 + 2 * delta_dphase3 + delta_dphase4) / 6,
        (delta_phase1 + 2 * delta_phase2 + 2 * delta_phase3 + delta_phase4) / 6,
    )



def check_vel_torch(vel):
    return not torch.any(torch.abs(vel)>1e-6)