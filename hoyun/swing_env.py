""" TODO
- fail된 노드들은 power와 mass를 0으로 만들고, 그대로 swing equation에 참여시킨다.
그러면 해당 노드의 acceleration을 계산할 때 mass로 나누어야 하기 때문에 zero division...?
+ 속도가 다시 threshold를 넘어가는 상황은 발생하지 않는가?

- fail된 노드의 gamma는 그대로 놔두는가?

- 현재는 deterministic한 action

- agent에게 넘기는 정보는 fail되기 직전의 power/mass/gamma + 어떤 노드가 fail 되는지
                   혹은 fail된 후 0으로 바뀐 power/mass + 어떤 노드가 fail 되었는지
"""

import functools
from typing import Any, Callable

import gym
import gym.spaces as spaces
import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.sparse
import solver
from graph.utils import edge_list_2_adjacency_matrix
from swing_data import SwingData

arr32 = npt.NDArray[np.float32]


class SwingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        swing_data: SwingData,
        dt: float = 1e-3,
        solver_name: str = "rk4",
        threshold: float = 1e-2,
        equilibrium_step: int = 1000,
        seed: int | None = None
    ) -> None:
        """
        swing_data: swing data
        dt: dt for swing simulation
        solver_name: which solver to solve swing equation
        e.g., RK1: "rk1",  "rk1_original", "rk1_sparse"
              RK2: "rk2",  "rk2_original", "rk2_sparse"
              RK4: "rk4",  "rk4_original", "rk4_sparse"
        threshold: whether to check if node is failed
        equilibrium_step: If reaching the iteration step, consider the system as equilibrium
        """
        # Random engine
        self.rng = np.random.default_rng(seed)

        # Network variables
        self.edge_list = swing_data["edge_list"]
        self.graph: nx.Graph = nx.from_edgelist(self.edge_list)
        self.num_nodes = self.graph.number_of_nodes()
        self.couplings = swing_data["coupling"]
        self.weighted_adjacency_matrix = edge_list_2_adjacency_matrix(
            self.edge_list, self.couplings
        )

        # swing variables
        self.initial_phase = swing_data["phase"]  # (N, )
        self.initial_dphase = swing_data["dphase"]  # (N, )
        self.initial_power = swing_data["power"]  # (N, )
        self.initial_gamma = swing_data["gamma"]  # (N, )
        self.initial_mass = swing_data["mass"]  # (N, )
        self.threshold = threshold
        self.equilibrium_step = equilibrium_step
        self.is_generator = self.initial_power > 0

        # swing solver
        self.step_solver: Callable[[arr32, arr32, arr32], tuple[arr32, arr32]]
        if "sparse" in solver_name:
            solver_module = solver.sparse
            self.weighted_adjacency_matrix = scipy.sparse.coo_matrix(
                self.weighted_adjacency_matrix
            )
            self.dt = dt
        elif "original" in solver_name:
            solver_module = solver.original
            self.dt = np.array(dt, dtype=np.float32)
        else:
            solver_module = solver.default
            self.dt = np.array(dt, dtype=np.float32)

        if "rk1" in solver_name:
            self.step_solver = functools.partial(
                solver_module.rk1,
                self.weighted_adjacency_matrix,
                dt=self.dt,
            )
        elif "rk2" in solver_name:
            self.step_solver = functools.partial(
                solver_module.rk2,
                self.weighted_adjacency_matrix,
                dt=self.dt,
            )
        elif "rk4" in solver_name:
            self.step_solver = functools.partial(
                solver_module.rk4,
                self.weighted_adjacency_matrix,
                dt=self.dt,
            )
        else:
            raise ValueError(f"No such solver: {solver_name}")

        # Initial states: steady state
        params = np.stack([self.initial_power, self.initial_gamma, self.initial_mass])
        while True:
            # Iterate few more times for mitigating numerical error during disk IO
            if self.is_stable(self.initial_dphase):
                break
            self.initial_phase, self.initial_dphase = self.step_solver(
                params, self.initial_phase, self.initial_dphase
            )
        self.initial_phase %= 2 * np.pi

        # Current state
        self.phase = self.initial_phase.copy()
        self.dphase = self.initial_dphase.copy()
        self.power = self.initial_power.copy()
        self.gamma = self.initial_gamma.copy()
        self.mass = self.initial_mass.copy()

        # Randomly fail single node as external perturbation
        self.failed = np.zeros(self.num_nodes, dtype=np.bool_)
        self.failed_at_this_step = np.zeros(self.num_nodes, dtype=np.bool_)
        self._randomly_fail_node()

        # RL variables
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.num_nodes,), dtype=np.float32
        )  # power rebalancing couplings for each node
        self.observation_space = spaces.Dict(
            {
                "phase": spaces.Box(
                    0.0, 2.0 * np.pi, (self.num_nodes,), dtype=np.float32
                ),
                "dphase": spaces.Box(
                    -np.inf, np.inf, (self.num_nodes,), dtype=np.float32
                ),
                "power": spaces.Box(
                    -np.inf, np.inf, (self.num_nodes,), dtype=np.float32
                ),
                "gamma": spaces.Box(0.0, np.inf, (self.num_nodes,), dtype=np.float32),
                "mass": spaces.Box(0.0, np.inf, (self.num_nodes,), dtype=np.float32),
                "step": spaces.Discrete(self.equilibrium_step),
                "failed_at_this_step": spaces.MultiDiscrete([2] * self.num_nodes),
            }
        )

    def step(
        self,
        action: arr32,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        action: (N, ), value bounded from -1 to 1. power rebalancing weigths for each node

        Return: tuple of observation, reward, terminated, info
        reward: number of failed nodes
        terminated: True if number of swing solver step is self.equilibrium
        """
        # Rebalance power
        self._rebalance_power(action)

        # Fail the node
        self._fail_nodes(self.failed_at_this_step)

        # Run swing equation until equilibrium / next failure
        step = 0
        params = np.stack([self.power, self.gamma, self.mass])
        for step in range(self.equilibrium_step):
            self.phase, self.dphase = self.step_solver(params, self.phase, self.dphase)

            # Get currently failed nodes whether they are failed before or not
            failed = self._get_failed_nodes(self.dphase)

            # Compare currently failed nodes with already failed nodes
            self.failed_at_this_step = ~self.failed * failed

            # If new nodes are failed at this step
            if self.failed_at_this_step.any():
                self.failed += self.failed_at_this_step
                break

        # Observe state
        observation: dict[str, Any] = {
            "phase": self.phase % (2 * np.pi),
            "dphase": self.dphase,
            "power": self.power,
            "gamma": self.gamma,
            "mass": self.mass,
            "step": step,
            "failed_at_this_step": self.failed_at_this_step,
        }
        reward = np.sum(self.dphase).item()
        terminated = step == self.equilibrium_step - 1

        return observation, reward, terminated, {}

    def reset(self) -> dict[str, Any]:
        """Reset environments and returns to the initial observation
        Initial observation: 0 steps until failure, all nodes are not failed
        """

        # Reset current states
        self.phase = self.initial_phase.copy()
        self.dphase = self.initial_dphase.copy()
        self.power = self.initial_power.copy()
        self.gamma = self.initial_gamma.copy()
        self.mass = self.initial_mass.copy()

        # Randomly fail single node as external perturbation
        self.failed = np.zeros(self.num_nodes, dtype=np.bool_)
        self.failed_at_this_step = np.zeros(self.num_nodes, dtype=np.bool_)
        self._randomly_fail_node()

        return {
            "phase": self.phase,
            "dphase": self.dphase,
            "power": self.power,
            "gamma": self.gamma,
            "mass": self.mass,
            "step": 0,
            "failed_at_this_step": self.failed_at_this_step,
        }

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            raise ValueError(f"No such rendering mode {mode}")

        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        fig, ax = plt.subplots()
        nx.draw(
            self.graph,
            ax=ax,
            node_size=100,
            node_color=self.phase,
            cmap="hsv",
            vmin=0.0,
            vmax=2.0 * np.pi,
        )
        sm = ScalarMappable(cmap="hsv", norm=Normalize(vmin=0.0, vmax=2.0 * np.pi))
        plt.colorbar(sm, ax=ax)
        fig.show()

    def _rebalance_power(self, action: arr32) -> None:
        # Normalize action from 0 to 1
        action = 0.5 * (action + 1.0)

        # Total fluctuation of power due to the failure
        power_fluctuation = np.sum(self.power[self.failed_at_this_step])

        # Get currently active generators
        active_generators = ~self.failed * self.is_generator

        # Do power rebalance for each active generators
        self.power[active_generators] += (
            action[active_generators] / action[active_generators].sum()
        ) * power_fluctuation

    def _fail_nodes(self, failed_nodes: npt.NDArray[np.bool_]) -> None:
        self.power[failed_nodes] = 0.0
        # self.mass[failed_nodes] = 0.0

    def _get_failed_nodes(self, dphase: arr32) -> npt.NDArray[np.bool_]:
        return dphase > self.threshold

    def _randomly_fail_node(self) -> None:
        failed_node = self.rng.choice(self.num_nodes)
        self.failed[failed_node] = True
        self.failed_at_this_step[failed_node] = True

    @staticmethod
    def is_stable(dphase: arr32, eps: float = 1e-4) -> bool:
        """Check if all dphase are smaller than eps"""
        return np.all(np.abs(dphase) < eps).item()
