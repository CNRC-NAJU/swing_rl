import warnings
from dataclasses import dataclass
from typing import Any, Literal, get_args

from config.distribution import DistributionConfig

TOPOLOGY = Literal["shk", "ba", "er", "rr", "complete"]


@dataclass(slots=True)
class SHKConfig:
    """shk (schultz-heitzh-kurths) parameters"""

    p: float = 0.2
    q: float = 0.3
    r: float = 0.3333
    s: float = 0.1
    initial: int = 1


@dataclass(slots=True)
class GraphConfig:
    _topology: TOPOLOGY = "shk"

    # Distribution of networks
    _num_nodes_distribution: DistributionConfig = DistributionConfig(
        name="uniform", low=10.0, high=10.0
    )
    _mean_degree_distribution: DistributionConfig = DistributionConfig(
        name="uniform", low=4.0, high=4.0
    )

    # SHK
    shk: SHKConfig = SHKConfig()

    def __post__init__(self) -> None:
        assert self.validate_topology(self._topology)
        assert self.validate_num_nodes_distribution(self._num_nodes_distribution)
        assert self.validate_mean_degree_distribution(self._mean_degree_distribution)

    def from_dict(self, config: dict[str, Any]) -> None:
        self.num_nodes_distribution = DistributionConfig(
            **config.pop("num_nodes_distribution")
        )
        self.mean_degree_distribution = DistributionConfig(
            **config.pop("mean_degree_distribution")
        )
        self.shk = SHKConfig(**config.pop("shk"))

        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    # --------------------- topology -------------------------
    @staticmethod
    def validate_topology(topology: TOPOLOGY) -> bool:
        return topology in get_args(TOPOLOGY)

    @property
    def topology(self) -> TOPOLOGY:
        return self._topology

    @topology.setter
    def topology(self, topology: TOPOLOGY) -> None:
        if not self.validate_topology(topology):
            warnings.warn("Invalid Topology. Ignore", stacklevel=2)
            return
        self._topology = topology

    # --------------------- distribution -------------------------
    @staticmethod
    def validate_num_nodes_distribution(distribution: DistributionConfig) -> bool:
        valid_min = distribution.low is not None and distribution.low > 0
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

    @staticmethod
    def validate_mean_degree_distribution(distribution: DistributionConfig) -> bool:
        valid_min = distribution.low is not None and distribution.low > 0
        valid_name = distribution.name in ["uniform", "normal"]

        return valid_min and valid_name

    @property
    def mean_degree_distribution(self) -> DistributionConfig:
        return self._mean_degree_distribution

    @mean_degree_distribution.setter
    def mean_degree_distribution(self, distribution: DistributionConfig) -> None:
        if not self.validate_mean_degree_distribution(distribution):
            warnings.warn("Invalid mean degree distribution. Ignore", stacklevel=2)
            return
        self._mean_degree_distribution = distribution
