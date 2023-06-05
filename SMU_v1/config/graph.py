import warnings
from dataclasses import dataclass
from typing import Any, Literal, get_args

from .distribution import DistributionConfig

_TOPOLOGY = Literal["shk", "ba", "er", "rr"]


@dataclass(slots=True)
class GraphConfig:
    # Distribution of networks
    mean_degree: float = 4.0
    _topology: _TOPOLOGY = "shk"
    _num_nodes_distribution: DistributionConfig = DistributionConfig(
        name="uniform", min=10.0, max=10.0
    )

    # shk (schultz-heitzh-kurths) parameters
    shk_p: float = 0.2
    shk_q: float = 0.3
    shk_r: float = 0.3333
    shk_s: float = 0.1
    shk_initial: int = 1

    def __post__init__(self) -> None:
        assert self.validate_topology(self._topology)
        assert self.validate_num_nodes_distribution(self._num_nodes_distribution)

    def from_dict(self, config: dict[str, Any]) -> None:
        self.num_nodes_distribution = DistributionConfig(
            **config.pop("num_nodes_distribution")
        )
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    # --------------------- topology -------------------------
    @staticmethod
    def validate_topology(topology: _TOPOLOGY) -> bool:
        return topology in get_args(_TOPOLOGY)

    @property
    def topology(self) -> _TOPOLOGY:
        return self._topology

    @topology.setter
    def topology(self, topology: _TOPOLOGY) -> None:
        if not self.validate_topology(topology):
            warnings.warn("Invalid Topology. Ignore", stacklevel=2)
            return
        self._topology = topology

    # --------------------- num nodes distribution -------------------------
    @staticmethod
    def validate_num_nodes_distribution(distribution: DistributionConfig) -> bool:
        valid_min = distribution.min is not None and distribution.min > 0
        valid_name = distribution.name in ["uniform", "normal"]

        return valid_min and valid_name

    @property
    def num_nodes_distribution(self) -> DistributionConfig:
        return self._num_nodes_distribution

    @num_nodes_distribution.setter
    def num_nodes_distribution(self, distribution: DistributionConfig) -> None:
        if not self.validate_num_nodes_distribution(distribution):
            warnings.warn("Invalid num nodes distribution. Ignore", stacklevel=2)
            return
        self._num_nodes_distribution = distribution


GRAPH_CONFIG = GraphConfig()
