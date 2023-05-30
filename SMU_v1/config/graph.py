from dataclasses import dataclass
from typing import Any, Literal

from .distribution import DistributionConfig


@dataclass(slots=True)
class GraphConfig:
    # Distribution of networks
    mean_degree: float = 4.0
    topology: Literal["shk", "ba", "er", "rr"] = "shk"
    num_nodes_distribution: DistributionConfig = DistributionConfig(
        name="uniform", min=10.0, max=10.0
    )

    # shk (schultz-heitzh-kurths) parameters
    shk_p: float = 0.2
    shk_q: float = 0.3
    shk_r: float = 0.3333
    shk_s: float = 0.1
    shk_initial: int = 1

    def __post__init__(self) -> None:
        assert self.topology in ["shk", "ba", "er", "rr"]
        assert self.num_nodes_distribution.name in ["uniform", "normal"]

    def from_dict(self, config: dict[str, Any]) -> None:
        self.num_nodes_distribution = DistributionConfig(
            **config.pop("num_nodes_distribution")
        )
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)


GRAPH_CONFIG = GraphConfig()
