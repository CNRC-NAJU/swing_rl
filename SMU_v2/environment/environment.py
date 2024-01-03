import warnings
from typing import Any, Literal, get_args

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import numpy.typing as npt
from config.grid import GRID_CONFIG, SwingConfig
from config.rl import OBSERVATION, RL_CONFIG, ObservationConfig
from smu_grid import Grid, NodeType, reset

from .reward import get_reward_ftn

arr32 = npt.NDArray[np.float32]
arr64 = npt.NDArray[np.float64]


class Environment(gym.Env):
    def __init__(
        self,
        grid: Grid,
        rng: np.random.Generator | int | None = None,
        verbose: Literal[0, 1, 2] = 0,
    ) -> None:
        """
        grid: Initial grid where the environment will start.
        rng: Random Number Generator or seed of RNG \\
        verbose
            0: no logging
            1: basic logging
            2: detailed logging. Recommended for debugging
        """
        if RL_CONFIG.reset.graph:
            warnings.warn(
                "Curretly, reset graph is not supported. Abort",
                stacklevel=2,
            )
            return

        super().__init__()
        self.verbose = verbose

        # Random engine
        if isinstance(rng, np.random.Generator):
            self.np_random = rng
        else:
            self.np_random = np.random.default_rng(rng)

        # Grid, steady state, perturbation variables
        self.grid = grid
        self.steady_phase: arr64
        self.steady_dphase: arr64
        self.marked: npt.NDArray[np.int64]

        # RL variables
        self.num_steps: int  # Current step inside the episode
        self.reward_ftn = get_reward_ftn(RL_CONFIG.reward)
        self.reward_failed_ftn = get_reward_ftn(RL_CONFIG.reward_failed)
        self.reward_failed_rebalance_ftn = get_reward_ftn(
            RL_CONFIG.reward_failed_rebalance
        )
        self.reward_failed_steady_ftn = get_reward_ftn(RL_CONFIG.reward_failed_steady)

        # Action, observation space
        self.action_space = self.set_action_space(self.grid.num_nodes)
        self.observation_space = self.set_observation_space(
            self.grid.num_nodes, self.grid.num_edges
        )

    def log(self, msg: str, level: int) -> None:
        """print the msg depending on it's report level. Lower level has higher priority"""
        if level > self.verbose:
            # Level is higher than verbose: ignore the message
            return
        print(msg)

    @staticmethod
    def find_steady_state(
        grid: Grid, config: SwingConfig | None = None, **kwargs
    ) -> tuple[arr64, arr64] | tuple[None, None]:
        """
        Find the steady state of grid. If not found, return None

        Args
        grid: target grid to find steady state
        config: swing configuration for finding steady state
        kwargs: passed to grid.run

        Return
        phase: [N, ]. Steady phase
        dphase: [N, ]. Steady dphase
        """
        if config is None:
            config = GRID_CONFIG.steady

        phases, dphases = grid.run(config=config, **kwargs)
        simulation_time = (len(phases) - 1) * config.dt

        # Successful to find steady state
        if simulation_time < config.max_time:
            return phases[-1], dphases[-1]
        else:
            return None, None

    def step(
        self, action: arr32
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        action: (N, 1), value bounded from -1 to 1. power rebalancing weights for generator, sink

        Return
        observation: element of observation_space
        reward: failed_reward/ number of failed nodes
        terminated: True if number of steps reached num_steps_per_episode
        truncated: True if the episode is finished before termination due to
                   unsuccessful rebalancing, failed nodes, ...
        info: Nothing
        """
        # Perturbate the node powers
        self.grid.perturbate(self.marked)

        # Pre-process action
        if RL_CONFIG.train_rebalance.strategy in ["directed", "deterministic"]:
            action = 0.5 * (action + 1.0)  # [-1, 1] -> [0, 1]
        action *= self.grid.is_generator + self.grid.is_sink  # Only controllable nodes
        self.log(f"Step: after pre-processing, action={np.round(action, 2)}", 2)

        if action.sum() == 0:
            # When given action is all zero, skip rebalancing and truncate
            balanced = False
        else:
            # Rebalance the grid according to the rebalance configuration, given action
            balanced = self.grid.rebalance(RL_CONFIG.train_rebalance, action)

        if not balanced:
            # Rebalance failed. Dummy dphases
            reward = self.reward_failed_rebalance_ftn(np.array([]), RL_CONFIG.swing.dt)
            self.log(f"Step: unsuccessful rebalancing. {reward=:.2e}", 2)
            return (
                self.observe(),  # observation
                reward,  # reward
                True,  # terminated
                False,  # truncated
                {},  # info
            )

        # Run swing equation, starting from steady state
        phases, dphases = self.grid.run(
            config=RL_CONFIG.swing, phase=self.steady_phase, dphase=self.steady_dphase
        )

        simulation_time = (len(phases) - 1) * RL_CONFIG.swing.dt
        if simulation_time < RL_CONFIG.swing.max_time:
            # Early stop the iteration: failed node exists
            # reward = self.reward_failed_ftn(dphases, RL_CONFIG.swing.dt)
            reward = self.reward_failed_ftn(dphases, simulation_time)
            terminated, truncated = True, False
            self.log(f"Step: failed node exists. {reward=:.2e}", 1)

        else:
            # No failed nodes: find steady state
            steady_phase, steady_dphase = self.find_steady_state(self.grid)
            if (steady_phase is None) or (steady_dphase is None):
                # Can't find steady state for current grid configuration
                reward = self.reward_failed_steady_ftn(dphases, RL_CONFIG.swing.dt)
                terminated, truncated = True, False
                self.log(f"Step: No steady state after rebalancing. {reward=:.2e}", 1)

            else:
                reward = self.reward_ftn(dphases, RL_CONFIG.swing.dt)
                self.steady_phase, self.steady_dphase = steady_phase, steady_dphase
                self.num_steps += 1

                terminated = False
                truncated = self.num_steps == RL_CONFIG.num_steps_per_episode
                self.log(f"Step: successfully finished rebalancing, {reward=:.2e}", 2)

        # Mark next perturbation if not truncated or terminated
        if not (terminated or truncated):
            self.marked = self.grid.mark_perturbations()
            self.log(f"Step: next perturbation will be {self.marked}", 2)

        return self.observe(), reward, terminated, truncated, {}

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environments and returns to the initial observation
        Initial observation: 0 steps until failure, all nodes are not failed

        Return
        observation: element of observation_space
        info: nothing
        """
        super().reset(seed=seed)  # self.np_random
        # Reset grid
        trial = 0
        while True:
            trial += 1
            grid = reset(self.grid, config=RL_CONFIG.reset)
            grid.turn_on()
            steady_phase, steady_dphase = self.find_steady_state(grid)
            if (steady_phase is not None) and (steady_dphase is not None):
                break
        self.grid = grid
        self.steady_phase = steady_phase
        self.steady_dphase = steady_dphase
        self.marked = self.grid.mark_perturbations()

        self.log(f"Reset: Found steady state after {trial} trials", 1)
        self.log(f"Reset: perturbation will be {self.marked}", 2)

        # Reset episode
        self.num_steps = 0

        return self.observe(), {}

    @staticmethod
    def set_action_space(num_nodes: int) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, (num_nodes,))

    @staticmethod
    def set_observation_space(
        num_nodes: int, num_edges: int, config: ObservationConfig | None = None
    ) -> spaces.Dict:
        if config is None:
            config = RL_CONFIG.observation

        observation_space: dict[str, spaces.Space[Any]] = {}
        empty_space = spaces.Box(0, 0, (0,))

        # 4 types: generator/renewable/consumer/sink
        observation_space["node_type"] = (
            spaces.Box(0, len(NodeType), (num_nodes,), dtype=np.int64)
            if config.node_type
            else empty_space
        )

        # Normalized phase: [-pi, pi)
        observation_space["phase"] = (
            spaces.Box(-np.pi, np.pi, (num_nodes,)) if config.phase else empty_space
        )

        # velocity, dphase: unlimited
        observation_space["dphase"] = (
            spaces.Box(-np.inf, np.inf, (num_nodes,)) if config.dphase else empty_space
        )

        # mass, gamma: always positive
        observation_space["mass"] = (
            spaces.Box(0, np.inf, (num_nodes,)) if config.mass else empty_space
        )
        observation_space["gamma"] = (
            spaces.Box(0, np.inf, (num_nodes,)) if config.mass else empty_space
        )

        # power: unlimited
        observation_space["power"] = (
            spaces.Box(-np.inf, np.inf, (num_nodes,)) if config.power else empty_space
        )

        # Active ratio: [0.0, 1.0]
        observation_space["active_ratio"] = (
            spaces.Box(0.0, 1.0, (num_nodes,)) if config.active_ratio else empty_space
        )

        # Perturbation: unlimited
        assert config.perturbation
        observation_space["perturbation"] = spaces.Box(-np.inf, np.inf, (num_nodes,))

        # Edge list
        assert config.edge_list
        observation_space["edge_list"] = spaces.Box(
            0, num_nodes - 1, (2, 2 * num_edges), dtype=np.int64
        )

        # Couling constants, edge attributes: always positive
        assert config.coupling
        observation_space["coupling"] = spaces.Box(0.0, np.inf, (2 * num_edges,))

        return spaces.Dict(observation_space)

    def observe(
        self, config: ObservationConfig | None = None
    ) -> dict[str, npt.NDArray[np.int64 | np.float32]]:
        def normalize_phase(phase: arr64) -> arr64:
            """Normalize phase into [-pi, pi)"""
            return (phase + np.pi) % (2 * np.pi) - np.pi

        if config is None:
            config = RL_CONFIG.observation

        observation: dict[str, npt.NDArray[np.int64 | np.float32]] = {
            obs: np.array([]) for obs in get_args(OBSERVATION)
        }
        if config.node_type:
            observation["node_type"] = np.array(
                [node_type.value for node_type in self.grid.node_types], dtype=np.int64
            )
        if config.phase:
            observation["phase"] = normalize_phase(self.steady_phase).astype(np.float32)
        if config.dphase:
            observation["dphase"] = self.steady_dphase.astype(np.float32)
        if config.mass:
            observation["mass"] = self.grid.masses.astype(np.float32)
        if config.gamma:
            observation["gamma"] = self.grid.gammas.astype(np.float32)
        if config.power:
            observation["power"] = self.grid.powers.astype(np.float32)
        if config.active_ratio:
            observation["active_ratio"] = self.grid.active_ratios.astype(np.float32)
        if config.perturbation:
            observation["perturbation"] = self.marked.astype(np.float32)
        if config.edge_list:
            observation["edge_list"] = self.grid.undirected_edge_list
        if config.coupling:
            observation["coupling"] = self.grid.undirected_couplings.astype(np.float32)

        return observation
