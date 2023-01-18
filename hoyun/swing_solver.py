from functools import partial
from typing import Callable, cast, overload

import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.sparse
import solver
import torch
from graph.utils import get_edge_list, get_weighted_adjacency_matrix

arr32 = npt.NDArray[np.float32]
arr64 = npt.NDArray[np.float64]
arr = arr32 | arr64
tensor = torch.Tensor
sparse = scipy.sparse.csr_matrix | scipy.sparse.coo_matrix | scipy.sparse.csc_matrix


def directed2undirected(
    edge_list: npt.NDArray[np.int64], device: torch.device | None = None
) -> torch.LongTensor:
    """Get directed edge list of shape (E, 2),
    Return undirected edge index of shape (2, E), for torch_geometric"""
    return torch.tensor(
        np.concatenate([edge_list, edge_list[:, [1, 0]]]), device=device
    ).T  # pyright: ignore


def step_solve_cpp(
    solver_name: str,
    edge_list: npt.NDArray[np.int64],
    weights: arr,
    phase: arr,
    dphase: arr,
    params: arr,
    dts: arr,
) -> arr:
    import shlex
    import string
    import subprocess
    from pathlib import Path

    SOLVER_DIR = Path(__file__).resolve().parent / "solver"
    HASH = "".join(s for s in np.random.choice(list(string.ascii_letters), 10))
    ARG_FILE = SOLVER_DIR / f"tmp_{HASH}.txt"
    EXECUTABLE = SOLVER_DIR / "simulation.out"

    precision = 32 if dts.dtype == np.float32 else 64
    digits = 6 if dts.dtype == np.float32 else 16

    # compile
    if not EXECUTABLE.exists():
        subprocess.run(
            shlex.split(
                f"g++ -O2 -flto -std=c++17 -o {EXECUTABLE} "
                f"{SOLVER_DIR}/cpp/main.cpp"
            )
        )

    # Change arguments to input of executable
    num_nodes, num_edges, num_steps = len(phase), len(edge_list), len(dts)
    with open(ARG_FILE, "w") as f:
        f.write("\n".join(f"{p:.{digits}f}" for p in phase) + "\n")
        f.write("\n".join(f"{p:.{digits}f}" for p in dphase) + "\n")
        f.write("\n".join(f"{p:.{digits}f}" for p in params[0]) + "\n")
        f.write("\n".join(f"{p:.{digits}f}" for p in params[1]) + "\n")
        f.write("\n".join(f"{p:.{digits}f}" for p in params[2]) + "\n")
        f.write(
            "\n".join(
                f"{node1}\n{node2}\n{weight:.{digits}f}"
                for (node1, node2), weight in zip(edge_list, weights)
            )
        )
        f.write("\n" + "\n".join(f"{dt:.{digits}f}" for dt in dts))

    # Run executable
    result = subprocess.check_output(
        shlex.split(
            f"{EXECUTABLE} {solver_name} {num_nodes} {num_edges} {num_steps} {precision} {ARG_FILE}"
        ),
        text=True,
    )
    ARG_FILE.unlink()

    trajectory = cast(arr, np.array(result.strip().split(), dtype=dts.dtype))
    return trajectory.reshape(-1, 2, num_nodes)


def step_solve(
    solver_name: str,
    weighted_adjacency_matrix: arr | sparse,
    phase: arr,
    dphase: arr,
    params: arr,
    dts: arr,
) -> list[arr]:
    if "sparse" in solver_name:
        solver_module = solver.sparse
        weighted_adjacency_matrix = scipy.sparse.coo_matrix(weighted_adjacency_matrix)
    elif "original" in solver_name:
        solver_module = solver.original
    else:
        solver_module = solver.default

    step_solver: Callable[[arr, arr, arr], tuple[arr, arr]]
    if "rk1" in solver_name:
        step_solver = partial(solver_module.rk1, weighted_adjacency_matrix, params)
    elif "rk2" in solver_name:
        step_solver = partial(solver_module.rk2, weighted_adjacency_matrix, params)
    elif "rk4" in solver_name:
        step_solver = partial(solver_module.rk4, weighted_adjacency_matrix, params)
    else:
        raise ValueError(f"No such solver: {solver_name}")

    trajectory = [cast(arr, np.stack([phase, dphase]))]
    for dt in dts:
        phase, dphase = step_solver(phase, dphase, np.array(dt, dtype=params.dtype))
        trajectory.append(cast(arr, np.stack([phase, dphase])))
    return trajectory


def step_solve_gpu(
    solver_name: str,
    weighted_adjacency_matrix: tensor,
    phase: tensor,
    dphase: tensor,
    params: tensor,
    dts: list[float],
) -> torch.Tensor:
    if "sparse" in solver_name:
        weighted_adjacency_matrix = weighted_adjacency_matrix.to_sparse_coo()

    step_solver: Callable[[tensor, tensor, float], tuple[tensor, tensor]]
    if "rk1" in solver_name:
        step_solver = partial(solver.gpu.rk1, weighted_adjacency_matrix, params)
    elif "rk2" in solver_name:
        step_solver = partial(solver.gpu.rk2, weighted_adjacency_matrix, params)
    elif "rk4" in solver_name:
        step_solver = partial(solver.gpu.rk4, weighted_adjacency_matrix, params)
    else:
        raise ValueError(f"No such solver: {solver_name}")

    trajectory = torch.zeros(
        (len(dts) + 1, 2, len(phase)), dtype=phase.dtype, device=phase.device
    )
    trajectory[0] = torch.stack([phase, dphase])
    for i, dt in enumerate(dts):
        phase, dphase = step_solver(phase, dphase, dt)
        trajectory[i + 1] = torch.stack([phase, dphase])
    return trajectory


def step_solve_gpu_scatter(
    solver_name: str,
    edge_list: torch.LongTensor,
    weights: tensor,
    phase: tensor,
    dphase: tensor,
    params: tensor,
    dts: list[float],
) -> torch.Tensor:
    step_solver: Callable[[tensor, tensor, float], tuple[tensor, tensor]]
    if "rk1" in solver_name:
        step_solver = partial(solver.gpu_scatter.rk1, edge_list, weights, params)
    elif "rk2" in solver_name:
        step_solver = partial(solver.gpu_scatter.rk2, edge_list, weights, params)
    elif "rk4" in solver_name:
        step_solver = partial(solver.gpu_scatter.rk4, edge_list, weights, params)
    else:
        raise ValueError(f"No such solver: {solver_name}")

    trajectory = torch.zeros(
        (len(dts) + 1, 2, len(phase)), dtype=phase.dtype, device=phase.device
    )
    trajectory[0] = torch.stack([phase, dphase])
    for i, dt in enumerate(dts):
        phase, dphase = step_solver(phase, dphase, dt)
        trajectory[i + 1] = torch.stack([phase, dphase])
    return trajectory


@overload
def solve(
    solver_name: str,
    graph: nx.Graph,
    weights: arr32,
    initial_phase: arr32,
    initial_dphase: arr32,
    params: arr32,
    dts: arr32,
    device: torch.device | None = None,
) -> arr32:
    ...


@overload
def solve(
    solver_name: str,
    graph: nx.Graph,
    weights: arr64,
    initial_phase: arr64,
    initial_dphase: arr64,
    params: arr64,
    dts: arr64,
    device: torch.device | None = None,
) -> arr64:
    ...


def solve(
    solver_name: str,
    graph: nx.Graph,
    weights: arr,
    initial_phase: arr,
    initial_dphase: arr,
    params: arr,
    dts: arr,
    device=None,
):
    """
    Solve swing equation:
    m_i * d^2 theta_i / dt^2 = P_i - gamma_i * d theta_i/dt + sum_j K_ij * A_ij * sin(theta_j-theta_i)

    solver_name: How to solve.
    e.g., RK4: "rk4", "rk4_cpp", "rk4_original", "rk4_sparse", "rk4_gpu", "rk4_gpu_sparse", "rk4_gpu_scatter"
    graph: graph that swing equation evolves on
    weights: (E, ) coupling constant of edges on graph
    initial_phase: (N, ) initial phase of nodes on graph, where N = graph.number_of_nodes()
    initial_dphase: (N, ) initial delta_phase (velocity) of nodes on graph
    params: (3, N) node features of power, gamma, mass
    dts: (S, ), dt for each time step
    device: When given, utilize gpu

    Return: (S+1, 2, N), (phase, dphase) of all nodes at each time step
    """
    if "cpp" in solver_name:
        trajectory = step_solve_cpp(
            solver_name,
            get_edge_list(graph),
            weights,
            initial_phase,
            initial_dphase,
            params,
            dts,
        )
        return trajectory

    elif "gpu" in solver_name:
        assert device is not None
        # Move to tensor
        initial_phase_torch = torch.tensor(initial_phase, device=device)
        initial_dphase_torch = torch.tensor(initial_dphase, device=device)
        params_torch = torch.tensor(params, device=device)
        dts_list = dts.tolist()

        if "scatter" in solver_name:
            edge_list = get_edge_list(graph)
            trajectory = step_solve_gpu_scatter(
                solver_name,
                directed2undirected(edge_list, device=device),
                torch.tensor(np.concatenate([weights, weights]), device=device),
                initial_phase_torch,
                initial_dphase_torch,
                params_torch,
                dts_list,
            )
        else:
            weighted_adjacency_matrix = get_weighted_adjacency_matrix(graph, weights)
            trajectory = step_solve_gpu(
                solver_name,
                torch.tensor(weighted_adjacency_matrix, device=device),
                initial_phase_torch,
                initial_dphase_torch,
                params_torch,
                dts_list,
            )
        return trajectory.cpu().numpy()

    else:
        weighted_adjacency_matrix = get_weighted_adjacency_matrix(graph, weights)
        trajectory = step_solve(
            solver_name,
            weighted_adjacency_matrix,
            initial_phase,
            initial_dphase,
            params,
            dts,
        )
        return np.array(trajectory)
