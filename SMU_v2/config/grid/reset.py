from dataclasses import dataclass


@dataclass(slots=True, frozen=True, kw_only=True)
class ResetConfig:
    """Whether to reset the componet of grid"""

    graph: bool = False
    coupling: bool = True
    node_type: bool = True
    node: bool = True

    def __post_init__(self) -> None:
        if self.graph:
            assert self.coupling and self.node_type and self.node

        if self.node_type:
            assert self.node
