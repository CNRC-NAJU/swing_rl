""" TODO
- reward
- observation space
- action space
- termination condition
"""

import copy
from typing import Any

import gym
import gym.spaces as spaces
import numpy as np
import numpy.typing as npt
from config import GraphConfig, RKConfig, RLConfig
from environment.grid import Grid
from environment.trajectory import is_failed, is_stable, normalize_phase
from graph.utils import get_graph
from solver.step_solver import get_step_solver

arr32 = npt.NDArray[np.float32]
Rng = np.random.Generator | int | None


class SwingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, rng: Rng = None) -> None:
        """
        swing_config: Cofiguration of swing environment
        dt: dt for swing simulation
        solver_name: which solver to solve swing equation
        e.g., RK1: "rk1",  "rk1_original", "rk1_sparse"
              RK2: "rk2",  "rk2_original", "rk2_sparse"
              RK4: "rk4",  "rk4_original", "rk4_sparse"
        threshold: whether to check if node is failed
        equilibrium_step: If reaching the iteration step, consider the system as equilibrium
        """
        # --------------- Random setup ------------
        if isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)

        # -------- Powergrid configuration ---------
        graph = get_graph(GraphConfig(), self.rng)
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()

        # Initial state: phase, dphase
        phase, dphase = None, None
        while phase is None or dphase is None:
            self.initial_grid = Grid(graph, rng=self.rng)
            phase, dphase = self.find_steady_state(self.initial_grid)
        self.initial_phase, self.initial_dphase = phase, dphase

        # Current state
        self.grid = copy.deepcopy(self.initial_grid)
        self.phase = self.initial_phase.copy()
        self.dphase = self.initial_dphase.copy()

        # Randomly fail single node as external perturbation
        self.marked = self.grid.mark_perturbation(RLConfig.num_pertubation)

        # RL variables
        if RLConfig.rebalance == "directed":
            self.rebalance = self.grid.rebalance_directed
        elif RLConfig.rebalance == "undirected":
            self.rebalance = self.grid.rebalance_undirected
        else:
            raise ValueError(f"No such rebalance policy: {RLConfig.rebalance}")

        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.grid.num_nodes,), dtype=np.float32
        )
        self.observation_space = spaces.Dict({})

    def reset_coupling(self, grid: Grid) -> Grid:
        """Update coupling constants of edges at entire grid"""
        coupling = grid.create_coupling(self.num_edges, self.rng)
        grid.set_coupling(coupling)
        return grid

    def find_steady_state(
        self, grid: Grid
    ) -> tuple[npt.NDArray[np.float32] | None, npt.NDArray[np.float32] | None]:
        """Find random initial state of given grid.
        Args
            grid: In which grid to find initial state
            rng: used for initializing random phase
        Return
            phase: [N, ] phase of each node in (-pi, pi]
            dphase: [N, ] dphase (angular frequency) of each node
        """
        # Initial random state
        phase = self.rng.uniform(-np.pi, np.pi, grid.num_nodes).astype(np.float32)
        dphase = np.zeros_like(phase)

        # Run swing equation
        step_solver = get_step_solver(
            RKConfig.name, grid.weighted_adjacency_matrix, grid.params, RKConfig.dt
        )
        # Run default simulation steps for finding stable state
        for _ in range(RLConfig.default_steady_steps):
            phase, dphase = step_solver(phase, dphase)
            # Iterate until reaching steady state
            if is_stable(dphase):
                break
        else:
            # could not find steady state: return None
            return None, None

        return normalize_phase(phase), dphase

    def step(self, action: arr32) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        action: (N, ), value bounded from -1 to 1. power rebalancing weigths for generator

        Return: tuple of observation, reward, terminated, info
        reward: number of failed nodes
        terminated: True if number of swing solver step is self.equilibrium
        """
        # Perturbate the node powers
        self.grid.perturbate(self.marked)

        # Rebalance power
        action *= self.grid.is_generator  # Only leave generator nodes
        self.rebalance(action)

        # Run swing equation until equilibrium / next failure
        step_solver = get_step_solver(
            RKConfig.name,
            self.grid.weighted_adjacency_matrix,
            self.grid.params,
            RKConfig.dt,
        )
        num_failed, simulation_step = 0, 0
        for simulation_step in range(RLConfig.equilibrium_steps):
            self.phase, self.dphase = step_solver(self.phase, self.dphase)

            # If any node got failed, break
            num_failed = is_failed(self.dphase)
            if num_failed:
                break

        if num_failed:
            reward = 10.0
        else:
            # No node failed until equilibrium steps: assume steady state
            phase, dphase = self.find_steady_state(self.grid)
            if phase is None or dphase is None:
                # Can't find steady state for current grid configuration
                reward = 10.0
            else:
                self.phase, self.dphase = phase, dphase

        # Mark next perturbation
        self.marked = self.grid.mark_perturbation(RLConfig.num_pertubation)

        # Observe state
        observation: dict[str, Any] = {}

        # Reward is number of failed nodes
        reward = 0.0

        # When reching equlibrium or all generators are failed
        terminated = simulation_step == RLConfig.equilibrium_steps - 1

        return observation, reward, terminated, {}

    def reset(self) -> dict[str, Any]:
        """Reset environments and returns to the initial observation
        Initial observation: 0 steps until failure, all nodes are not failed
        """
        if RLConfig.reset_graph:
            graph = get_graph(GraphConfig(), self.rng)
            self.num_nodes = graph.number_of_nodes()
            self.num_edges = graph.number_of_edges()
            self.initial_grid = Grid(graph, rng=self.rng)
        elif RLConfig.reset_coupling:
            new_coupling = self.initial_grid.create_coupling(self.num_edges, self.rng)
            self.initial_grid.set_coupling(new_coupling)

        if RLConfig.reset_node_type:
            new_nodes = self.initial_grid.create_nodes(self.num_nodes, rng=self.rng)
            self.initial_grid.set_nodes(new_nodes)
        elif RLConfig.reset_node:
            node_types = self.initial_grid.node_types
            new_nodes = self.initial_grid.create_nodes(
                self.num_nodes, node_types, self.rng
            )
            self.initial_grid.set_nodes(new_nodes)

        # If initial states are changed, re-calculate them
        if RLConfig.reset_coupling or RLConfig.reset_node:
            graph = self.initial_grid.graph
            phase, dphase = None, None
            while phase is None or dphase is None:
                self.initial_grid = Grid(graph, rng=self.rng)
                phase, dphase = self.find_steady_state(self.initial_grid)
            self.initial_phase, self.initial_dphase = phase, dphase

        # Reset current states
        self.grid = copy.deepcopy(self.initial_grid)
        self.phase = self.initial_phase.copy()
        self.dphase = self.initial_dphase.copy()

        # Randomly fail single node as external perturbation
        self.marked = self.grid.mark_perturbation(RLConfig.num_pertubation)

        return {
            "phase": self.phase,
            "dphase": self.dphase,
            "params": self.grid.params,  # (3, N)
            "step": 0,
            "perturbated": self.marked,
        }
