from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .distribution import DistributionConfig
from .singleton import Singleton


@dataclass
class GraphConfig(metaclass=Singleton):
    topology: str = "shk"

    # Distribution of network size
    num_nodes_distribution: DistributionConfig = DistributionConfig(
        name="uniform", min=10.0, max=10.0
    )

    mean_degree: float = 4.0

    # shk parameters
    shk_p: float = 0.2
    shk_q: float = 0.3
    shk_r: float = 0.3333
    shk_s: float = 0.1
    shk_initial: int = 1

    def __post__init__(self) -> None:
        assert self.topology in ["shk", "ba", "er", "rr"]

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GraphConfig:
        distribution = DistributionConfig(**config.pop("num_nodes_distribution"))

        graph = cls(num_nodes_distribution=distribution)
        for key, value in config.items():
            setattr(graph, key, value)

        return graph
