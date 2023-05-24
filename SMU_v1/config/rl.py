import warnings
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class RLConfig:
    # Number of perturbated nodes at each steps
    num_pertubation: int = 1

    # simulation time
    steady_time: float = 20
    equilibrium_time: float = 1

    # Stability of node
    stable_threshold: float = 1e-4
    fail_threshold: float = 1e-2

    # Rebalancing policy
    rebalance: Literal["directed", "undirected", "deterministic"] = "directed"
    max_rebalance: int = 1000

    # Reset range
    reset_graph: bool = False
    reset_coupling: bool = False
    reset_node_type: bool = False
    reset_node: bool = True

    # Reward
    reward: Literal[
        "area", "slope", "weighted_area", "threshold_area", "weighted_threshold_area"
    ] = "weighted_threshold_area"
    failed_scale: float = 1.0

    # Episode
    num_steps_per_episode: int = 1000

    def __post_init__(self) -> None:
        assert self.rebalance in ["directed", "undirected"]
        if self.rebalance == "undirected":
            warnings.warn("Undirected rebalancing strategy may endup Error")

        if self.reset_graph:
            assert (
                self.reset_coupling
            ), f"When reset graph, you also need to reset coupling"
            assert (
                self.reset_node_type
            ), f"When reset graph, you also need to reset node_type"
            assert (
                self.reset_node
            ), f"When reset graph, you also need to reset entire nodes"

        if self.reset_node_type:
            assert (
                self.reset_node
            ), f"When reset node type, you also need to reset entire nodes"

        assert self.reward in [
            "area",
            "slope",
            "weighted_area",
            "threshold_area",
            "weighted_threshold_area",
        ]

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)


RL_CONFIG = RLConfig()
