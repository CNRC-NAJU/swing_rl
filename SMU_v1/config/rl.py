import warnings
from dataclasses import dataclass
from typing import Any, Literal, get_args

_REBALANCE = Literal["directed", "undirected", "deterministic"]
_REWARD = Literal[
    "area", "slope", "weighted_area", "threshold_area", "weighted_threshold_area"
]


@dataclass(slots=True)
class RLConfig:
    # Number of perturbated nodes at each steps
    num_pertubation: int = 1

    # simulation time
    steady_time: float = 20.0
    equilibrium_time: float = 1.0

    # Stability of node
    stable_threshold: float = 1e-4
    fail_threshold: float = 1e-2

    # Rebalancing policy
    _rebalance: _REBALANCE = "directed"
    max_rebalance: int = 1000

    # Reset range
    _reset_graph: bool = False
    _reset_coupling: bool = False
    _reset_node_type: bool = False
    _reset_node: bool = True

    # Reward
    _reward: _REWARD = "weighted_threshold_area"
    failed_scale: float = 1.0

    # Episode
    num_steps_per_episode: int = 1000

    def __post_init__(self) -> None:
        assert self.validate_rebalance(self._rebalance)
        assert self.validate_reward(self._reward)
        assert self.validate_reset(
            self._reset_graph,
            self._reset_coupling,
            self._reset_node_type,
            self._reset_node,
        )

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    # --------------------- Rebalance -------------------------
    @staticmethod
    def validate_rebalance(rebalance: _REBALANCE) -> bool:
        if rebalance == "undirected":
            warnings.warn(
                "Undirected rebalancing strategy may endup Error", stacklevel=2
            )
        return rebalance in get_args(_REBALANCE)

    @property
    def rebalance(self) -> _REBALANCE:
        return self._rebalance

    @rebalance.setter
    def rebalance(self, value: _REBALANCE) -> None:
        if not self.validate_rebalance(value):
            warnings.warn("Invalid initial rebalance strategy. Ignore", stacklevel=2)
            return
        self._rebalance = value

    # --------------------- Reward -------------------------
    @staticmethod
    def validate_reward(reward: _REWARD) -> bool:
        return reward in get_args(_REWARD)

    @property
    def reward(self) -> _REWARD:
        return self._reward

    @reward.setter
    def reward(self, value: _REWARD) -> None:
        if not self.validate_reward(value):
            warnings.warn("Invalid reward. Ignore", stacklevel=2)
            return
        self._reward = value

    # --------------------- Reset -------------------------
    @staticmethod
    def validate_reset(
        graph: bool, coupling: bool, node_type: bool, node: bool
    ) -> bool:
        if graph:
            return coupling and node_type and node

        if node_type:
            return node

        return True

    @property
    def reset_graph(self) -> bool:
        return self._reset_graph

    @property
    def reset_coupling(self) -> bool:
        return self._reset_coupling

    @property
    def reset_node_type(self) -> bool:
        return self._reset_node_type

    @property
    def reset_node(self) -> bool:
        return self._reset_node

    @reset_graph.setter
    def reset_graph(self, value: bool) -> None:
        if value:
            warnings.warn(
                "Reset graph: also reset coupling/node_type/node", stacklevel=2
            )
            self._reset_coupling = True
            self._reset_node_type = True
            self._reset_node = True
        self._reset_graph = value

    @reset_coupling.setter
    def reset_coupling(self, value: bool) -> None:
        self._reset_coupling = value

    @reset_node_type.setter
    def reset_node_type(self, value: bool) -> None:
        if value:
            warnings.warn("Reset node type: also reset node", stacklevel=2)
            self._reset_node = True
        self._reset_node_type = value

    @reset_node.setter
    def reset_node(self, value: bool) -> None:
        self._reset_node = value


RL_CONFIG = RLConfig()
