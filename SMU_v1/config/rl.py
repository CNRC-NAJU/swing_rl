from dataclasses import dataclass


@dataclass
class RLConfig:
    # Number of perturbated nodes at each steps
    num_pertubation: int = 1

    # Number of simulation steps
    default_steady_steps: int = 10000
    equilibrium_steps: int = 1000

    # Stability of node
    stable_threshold: float = 1e-4
    fail_threshold: float = 1e-2

    # Rebalancing policy
    rebalance: str = "directed"

    # Reset range
    reset_graph: bool = False
    reset_coupling: bool = False
    reset_node_type: bool = False
    reset_node: bool = False

    def __post_init__(self) -> None:
        assert self.rebalance in ["directed", "undirected"]

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
