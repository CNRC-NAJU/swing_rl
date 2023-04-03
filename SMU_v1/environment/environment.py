import copy
from typing import Any

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import numpy.typing as npt
from config import ObservationConfig, RLConfig, SwingConfig

from .graph.utils import get_graph
from .grid import Grid
from .reward import get_reward_ftn, reward_failed
from .swing.solver import swing_solver
from .trajectory import is_failed, is_stable, normalize_phase

arr32 = npt.NDArray[np.float32]
Rng = np.random.Generator | int | None


class Environment(gym.Env):
    def __init__(self, rng: Rng = None) -> None:
        # --------------- Random setup ------------
        if isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)
        self.np_random = self.rng

        # -------- Powergrid configuration ---------
        graph = get_graph(self.rng)
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()

        # States
        phase, dphase = None, None
        while phase is None or dphase is None:
            self.initial_grid = Grid(graph, rng=self.rng)
            phase, dphase = self.find_steady_state(self.initial_grid)
        self.initial_phase, self.initial_dphase = phase, dphase
        self.grid = copy.deepcopy(self.initial_grid)
        self.phase = self.initial_phase.copy()
        self.dphase = self.initial_dphase.copy()

        # Randomly fail single node as external perturbation
        self.marked = self.grid.mark_perturbation(RLConfig.num_pertubation)

        # Reward
        self.reward = get_reward_ftn()

        # Rebalance strategy
        if RLConfig.rebalance == "directed":
            self.rebalance = self.grid.rebalance_directed
        elif RLConfig.rebalance == "undirected":
            self.rebalance = self.grid.rebalance_undirected
        else:
            raise ValueError(f"No such rebalance policy: {RLConfig.rebalance}")

        # RL variables
        self.steps_per_episode = 0
        self.action_space = self.set_action_space()
        self.observation_space = self.set_observation_space()

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
        phase = np.zeros(grid.num_nodes, dtype=np.float32)
        dphase = np.zeros_like(phase)

        # Run swing equation
        solver = swing_solver(grid.weighted_adjacency_matrix, grid.params)
        # Run default simulation steps for finding stable state
        time, dt = 0.0, np.array(SwingConfig.dt, dtype=np.float32)
        while time < RLConfig.steady_time:
            time += dt
            phase, dphase = solver(phase, dphase)
            if is_stable(dphase):
                return phase, dphase

        # could not find steady state: return None
        return None, None

    def step(
        self, action: arr32
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        action: (N, 1), value bounded from -1 to 1. power rebalancing weigths for generator

        Return: tuple of observation, reward, terminated, info
        reward: number of failed nodes
        terminated: True if number of swing solver step is self.equilibrium
        truncated: True if the episode is finished before termination
        info: Nothing
        """
        # Perturbate the node powers
        self.grid.perturbate(self.marked)

        # Rebalance power
        action *= self.grid.is_generator  # Only leave generator nodes
        self.rebalance(action)

        # Run swing equation until equilibrium / next failure
        solver = swing_solver(self.grid.weighted_adjacency_matrix, self.grid.params)
        time, num_failed, simulation_step = np.array(0.0, np.float32), 0, 0
        times: list[arr32] = []
        phases: list[arr32] = []
        dphases: list[arr32] = []
        dt = np.array(SwingConfig.dt, dtype=np.float32)
        while time < RLConfig.equilibrium_time:
            self.phase, self.dphase = solver(self.phase, self.dphase)
            time += dt

            # If any node got failed, break
            num_failed = is_failed(self.dphase)
            if num_failed:
                break

            # Store trajectory
            times.append(time)
            phases.append(self.phase.copy())
            dphases.append(self.dphase.copy())

        if num_failed:
            reward = reward_failed(num_failed, time.item())
            terminated, truncated = False, True
        else:
            # No node failed until equilibrium steps: assume steady state
            phase, dphase = self.find_steady_state(self.grid)
            if phase is None or dphase is None:
                # Can't find steady state for current grid configuration
                reward = reward_failed(num_failed=1, time=time.item())
                terminated, truncated = False, True
            else:
                self.phase, self.dphase = phase, dphase
                self.steps_per_episode += 1

                reward = self.reward(np.stack(dphases), np.stack(times))
                terminated = self.steps_per_episode == RLConfig.steps_per_episode
                truncated = False

        # Mark next perturbation
        self.marked = self.grid.mark_perturbation(RLConfig.num_pertubation)

        # Observe state
        observation = self.observe()

        return observation, reward, terminated, truncated, {}

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environments and returns to the initial observation
        Initial observation: 0 steps until failure, all nodes are not failed
        """
        if RLConfig.reset_graph:
            graph = get_graph(self.rng)
            self.num_nodes = graph.number_of_nodes()
            self.num_edges = graph.number_of_edges()
            self.action_space = self.set_action_space()
            self.observation_space = self.set_observation_space()
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
        self.steps_per_episode = 0
        self.grid = copy.deepcopy(self.initial_grid)
        self.phase = self.initial_phase.copy()
        self.dphase = self.initial_dphase.copy()

        # Randomly fail single node as external perturbation
        self.marked = self.grid.mark_perturbation(RLConfig.num_pertubation)

        return self.observe(), {}

    def set_action_space(self) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, (self.num_nodes,))

    def set_observation_space(self) -> spaces.Dict:
        observation_space: dict[str, spaces.Space] = {}

        if ObservationConfig.node_type:
            # 3 types: generator/renewable/consumer
            observation_space["node_type"] = spaces.Discrete(3)
        if ObservationConfig.phase:
            # Normalized angle, (-pi, pi]
            observation_space["phase"] = spaces.Box(-np.pi, np.pi, (self.num_nodes,))
        if ObservationConfig.dphase:
            # angular velocity: unlimited
            observation_space["dphase"] = spaces.Box(-np.inf, np.inf, (self.num_nodes,))
        if ObservationConfig.mass:
            observation_space["mass"] = spaces.Box(0, np.inf, (self.num_nodes,))
        if ObservationConfig.gamma:
            observation_space["gamma"] = spaces.Box(0, np.inf, (self.num_nodes,))
        if ObservationConfig.power:
            observation_space["power"] = spaces.Box(
                -np.inf, np.inf, (self.num_nodes,), dtype=np.int64
            )
        if ObservationConfig.active_ratio:
            observation_space["active_ratio"] = spaces.Box(-1.0, 1.0, (self.num_nodes,))
        if ObservationConfig.perturbation:
            observation_space["perturbation"] = spaces.Box(
                -1.0, 1.0, (self.num_nodes,), dtype=np.int64
            )
        if ObservationConfig.edge_list:
            observation_space["edge_list"] = spaces.Box(
                0, self.num_nodes - 1, (2, 2 * self.num_edges), dtype=np.int64
            )
        if ObservationConfig.coupling:
            observation_space["coupling"] = spaces.Box(
                0.0, np.inf, (2 * self.num_edges,)
            )

        return spaces.Dict(observation_space)

    def observe(self) -> dict[str, Any]:
        observation: dict[str, Any] = {}

        if ObservationConfig.node_type:
            observation["node_type"] = np.array(
                [node_type.value for node_type in self.grid.node_types], dtype=np.int64
            )
        if ObservationConfig.phase:
            observation["phase"] = normalize_phase(self.phase)
        if ObservationConfig.dphase:
            observation["dphase"] = self.dphase
        if ObservationConfig.mass:
            observation["mass"] = self.grid.masses
        if ObservationConfig.gamma:
            observation["gamma"] = self.grid.gammas
        if ObservationConfig.power:
            observation["power"] = self.grid.powers
        if ObservationConfig.active_ratio:
            observation["active_ratio"] = self.grid.active_ratios
        if ObservationConfig.perturbation:
            observation["perturbation"] = self.marked
        if ObservationConfig.edge_list:
            observation["edge_list"] = self.grid.edge_list
        if ObservationConfig.coupling:
            observation["coupling"] = self.grid.coupling

        return observation
