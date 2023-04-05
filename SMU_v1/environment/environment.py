import copy
from functools import partial
from typing import Any, Callable

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import numpy.typing as npt
from config import ObservationConfig, RLConfig, SwingConfig

from .grid import Grid
from .reward import get_reward_ftn, reward_failed
from .swing.solver import swing_solver
from .trajectory import is_failed, is_stable, normalize_phase

Rng = np.random.Generator | int | None
DTYPE = SwingConfig().dtype


class Environment(gym.Env):
    def __init__(self, grid: Grid, rng: Rng = None) -> None:
        # Random engine
        if isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)
        self.np_random = self.rng

        # Variables that will not change over steps
        # They might change for new episode depending on reset configuration
        while True:
            steady_phase, steady_dphase = self.find_steady_state(grid)
            if steady_phase is not None and steady_dphase is not None:
                break
            print("Couldn't find steady state")
            grid = self.reset_grid(grid)
        self.initial_grid = grid
        self.initial_steady_phase = steady_phase
        self.initial_steady_dphase = steady_dphase

        # RL variables: action/observation spaces, reward functions
        self.reward = get_reward_ftn()
        self.action_space = self.set_action_space(grid.num_nodes)
        self.observation_space = self.set_observation_space(
            grid.num_nodes, grid.num_edges
        )

        # States that will change over each steps
        self.grid = copy.deepcopy(self.initial_grid)
        self.steady_phase = self.initial_steady_dphase.copy()
        self.steady_dphase = self.initial_steady_dphase.copy()
        self.marked = self.grid.mark_perturbation(RLConfig.num_pertubation)
        self.num_steps = 0

    @property
    def rebalance(self) -> Callable[[npt.NDArray[np.float32]], bool]:
        rl_config = RLConfig()
        if rl_config.rebalance == "directed":
            return partial(
                self.grid.rebalance_directed, max_trial=rl_config.max_rebalance
            )
        elif rl_config.rebalance == "undirected":
            return partial(
                self.grid.rebalance_undirected, max_trial=rl_config.max_rebalance
            )
        else:
            raise ValueError(f"No such rebalance policy: {rl_config.rebalance}")

    def reset_grid(self, grid: Grid) -> Grid:
        """Reset grid according to reset level"""
        rl_config = RLConfig()
        if rl_config.reset_graph:
            # If reset graph, also need to update RL spaces
            grid.reset_graph()
            return grid

        if rl_config.reset_coupling:
            # Reset coupling
            grid.reset_coupling()

        if rl_config.reset_node_type:
            # Reset node type: node will automatically reset
            grid.reset_node_types()
        elif rl_config.reset_node:
            # Only reset node
            grid.reset_nodes()
        return grid

    @staticmethod
    def find_steady_state(
        grid: Grid, rng: Rng = None
    ) -> tuple[npt.NDArray[DTYPE] | None, npt.NDArray[DTYPE] | None]:
        """Find random initial state of given grid.
        Args
            grid: In which grid to find initial state
            rng: used for initializing random phase
        Return
            phase: [N, ] phase of each node in (-pi, pi]
            dphase: [N, ] dphase (angular frequency) of each node
        """
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # Initial random state
        phase = np.zeros(grid.num_nodes, dtype=DTYPE)
        dphase = np.zeros_like(phase)

        # Run swing equation until reaching steady time
        time = 0.0
        solver = swing_solver(grid.weighted_adjacency_matrix, grid.params)
        while time < RLConfig.steady_time:
            time += SwingConfig._dt
            phase, dphase = solver(phase=phase, dphase=dphase)
            print(time, np.abs(dphase).max())
            if is_stable(dphase):
                return phase, dphase

        # could not find steady state: return None
        return None, None

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        action: (N, 1), value bounded from -1 to 1. power rebalancing weigths for generator

        Return: tuple of observation, reward, terminated, info
        reward: number of failed nodes
        terminated: True if number of swing solver step is self.equilibrium
        truncated: True if the episode is finished before termination
        info: Nothing
        """
        rl_config, swing_config = RLConfig(), SwingConfig()

        # Perturbate the node powers
        self.grid.perturbate(self.marked)

        # Rebalance power
        action *= self.grid.is_generator  # Only leave generator nodes
        balanced = self.rebalance(action)
        if not balanced:
            return (
                self.observe(),
                reward_failed(self.grid.num_nodes, 0),
                False,
                True,
                {},
            )

        # Containers to store trajectory
        times: list[float] = []
        phases: list[npt.NDArray[DTYPE]] = []
        dphases: list[npt.NDArray[DTYPE]] = []

        # Run swing equation until equilibrium / next failure
        solver = swing_solver(self.grid.weighted_adjacency_matrix, self.grid.params)
        phase, dphase = self.steady_phase.copy(), self.steady_dphase.copy()
        time, num_failed = 0.0, 0
        while time < rl_config.equilibrium_time:
            phase, dphase = solver(phase=phase, dphase=dphase)
            time += swing_config._dt

            # If any node got failed, break
            num_failed = is_failed(dphase)
            if num_failed:
                break

            # Store trajectory
            times.append(time)
            phases.append(phase.copy())
            dphases.append(dphase.copy())

        if num_failed:
            reward = reward_failed(num_failed, time)
            terminated, truncated = False, True
        else:
            # No node failed until equilibrium steps: assume steady state
            steady_phase, steady_dphase = self.find_steady_state(self.grid)
            if steady_phase is None or steady_dphase is None:
                # Can't find steady state for current grid configuration
                reward = reward_failed(num_failed=1, time=time)
                terminated, truncated = False, True
            else:
                self.steady_phase, self.steady_dphase = steady_phase, steady_dphase
                self.num_steps += 1

                reward = self.reward(np.stack(dphases), np.array(times, dtype=DTYPE))
                terminated = self.num_steps == rl_config.num_steps_per_episode
                truncated = False

        # Mark next perturbation
        self.marked = self.grid.mark_perturbation(rl_config.num_pertubation)

        # Observe state
        observation = self.observe()

        return observation, reward, terminated, truncated, {}

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environments and returns to the initial observation
        Initial observation: 0 steps until failure, all nodes are not failed
        """
        rl_config = RLConfig()

        # If reset grid, re-calculate initial state
        if rl_config.reset_coupling or rl_config.reset_node:
            while True:
                self.initial_grid = self.reset_grid(self.initial_grid)
                steady_phase, steady_dphase = self.find_steady_state(self.initial_grid)
                if steady_phase is not None and steady_dphase is not None:
                    break
                print("Couldn't find steady state")
            self.initial_steady_phase = steady_phase
            self.initial_steady_dphase = steady_dphase
            self.action_space = self.set_action_space(self.initial_grid.num_nodes)
            self.observation_space = self.set_observation_space(
                self.initial_grid.num_nodes, self.initial_grid.num_edges
            )

        # Update state for new episode
        self.grid = copy.deepcopy(self.initial_grid)
        self.steady_phase = self.initial_steady_dphase.copy()
        self.steady_dphase = self.initial_steady_dphase.copy()
        self.num_steps = 0

        # Randomly fail single node as external perturbation
        self.marked = self.grid.mark_perturbation(rl_config.num_pertubation)

        return self.observe(), {}

    @staticmethod
    def set_action_space(num_nodes: int) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, (num_nodes,))

    @staticmethod
    def set_observation_space(num_nodes: int, num_edges: int) -> spaces.Dict:
        observation_config = ObservationConfig()
        observation_space: dict[str, spaces.Space] = {}

        if observation_config.node_type:
            # 3 types: generator/renewable/consumer
            observation_space["node_type"] = spaces.Discrete(3)
        if observation_config.phase:
            # Normalized angle, (-pi, pi]
            observation_space["phase"] = spaces.Box(-np.pi, np.pi, (num_nodes,))
        if observation_config.dphase:
            # angular velocity: unlimited
            observation_space["dphase"] = spaces.Box(-np.inf, np.inf, (num_nodes,))
        if observation_config.mass:
            observation_space["mass"] = spaces.Box(0, np.inf, (num_nodes,))
        if observation_config.gamma:
            observation_space["gamma"] = spaces.Box(0, np.inf, (num_nodes,))
        if observation_config.power:
            observation_space["power"] = spaces.Box(
                -np.inf, np.inf, (num_nodes,), dtype=np.int64
            )
        if observation_config.active_ratio:
            observation_space["active_ratio"] = spaces.Box(-1.0, 1.0, (num_nodes,))
        if observation_config.perturbation:
            observation_space["perturbation"] = spaces.Box(
                -1.0, 1.0, (num_nodes,), dtype=np.int64
            )
        if observation_config.edge_list:
            observation_space["edge_list"] = spaces.Box(
                0, num_nodes - 1, (2, 2 * num_edges), dtype=np.int64
            )
        if observation_config.coupling:
            observation_space["coupling"] = spaces.Box(0.0, np.inf, (2 * num_edges,))

        return spaces.Dict(observation_space)

    def observe(self) -> dict[str, Any]:
        observation_config = ObservationConfig()
        observation: dict[str, Any] = {}

        if observation_config.node_type:
            observation["node_type"] = np.array(
                [node_type.value for node_type in self.grid.node_types], dtype=np.int64
            )
        if observation_config.phase:
            observation["phase"] = normalize_phase(
                self.steady_phase.astype(np.float32, copy=False)
            )
        if observation_config.dphase:
            observation["dphase"] = self.steady_dphase.astype(np.float32, copy=False)
        if observation_config.mass:
            observation["mass"] = self.grid.masses.astype(np.float32, copy=False)
        if observation_config.gamma:
            observation["gamma"] = self.grid.gammas.astype(np.float32, copy=False)
        if observation_config.power:
            observation["power"] = self.grid.powers.astype(np.float32, copy=False)
        if observation_config.active_ratio:
            observation["active_ratio"] = self.grid.active_ratios
        if observation_config.perturbation:
            observation["perturbation"] = self.marked
        if observation_config.edge_list:
            observation["edge_list"] = self.grid.edge_list
        if observation_config.coupling:
            observation["coupling"] = self.grid.coupling

        return observation
