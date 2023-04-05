from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .distribution import DistributionConfig
from .singleton import Singleton


@dataclass
class GridConfig(metaclass=Singleton):
    # Distribution of couplings of each nodes
    coupling_distribution: DistributionConfig = DistributionConfig(
        name="uniform",
        min=10.0,
        max=10.0,
    )

    # Number ratio: generator + renewable + consumer = 1.0
    generator_num_ratio: float = 0.4
    renewable_num_ratio: float = 0.2
    consumer_num_ratio: float = 0.4

    # Power capacity ratio
    generator_spare: float = 1.1  # (generator capacity) = spare * (consumer capacity)
    source_ratio: float = 4.0  # (renewable capacity) = (generator capacity) / ratio

    # Initial activeness
    initial_active_ratio: float = 0.5
    initial_rebalance: str = "directed"
    initial_max_rebalance: int = 1000

    def __post_init__(self) -> None:
        assert (
            self.generator_num_ratio
            + self.renewable_num_ratio
            + self.consumer_num_ratio
            == 1.0
        )

        assert self.initial_rebalance in ["directed", "undirected"]

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GridConfig:
        distribution = DistributionConfig(**config.pop("coupling_distribution"))

        grid = cls(coupling_distribution=distribution)
        for key, value in config.items():
            setattr(grid, key, value)

        return grid
