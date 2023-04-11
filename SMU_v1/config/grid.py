from dataclasses import dataclass
from typing import Any

import numpy as np

from .distribution import DistributionConfig


@dataclass
class GridConfig:
    # Distribution of couplings of each nodes
    coupling_distribution: DistributionConfig = DistributionConfig(
        name="uniform",
        min=10.0,
        max=10.0,
    )

    # Number ratio: generator + renewable + consumer + controllable consumer = 1.0
    generator_num_ratio: float = 0.4
    renewable_num_ratio: float = 0.2
    consumer_num_ratio: float = 0.3
    controllable_consumer_num_ratio: float = 0.1

    # Distribution of nodes
    generator_capacity_distribution: DistributionConfig = DistributionConfig(
        name="uniform_wo_avg", delta=4.0
    )
    renewable_capacity_distribution: DistributionConfig = DistributionConfig(
        name="uniform_wo_avg", delta=4.0
    )
    renewable_mass_distribution: DistributionConfig = DistributionConfig(
        name="uniform", min=0.1, max=0.1
    )
    consumer_max_units_distribution: DistributionConfig = DistributionConfig(
        name="uniform", min=10.0, max=10.0
    )
    controllable_consumer_capacity_distribution: DistributionConfig = (
        DistributionConfig(name="uniform_wo_avg", delta=4.0)
    )

    # Power capacity ratio
    generator_spare: float = 1.1  # (generator capacity) = spare * (consumer capacity)
    source_ratio: float = 4.0  # (renewable capacity) = (generator capacity) / ratio
    controllable_consumer_spare: float = (
        1.1  # (controllable consumer capacity) = spare * (renewable capacity)
    )

    # Initial activeness
    initial_active_ratio: float = 0.5
    initial_rebalance: str = "directed"
    initial_max_rebalance: int = 1000

    def __post_init__(self) -> None:
        assert np.isclose(
            self.generator_num_ratio
            + self.renewable_num_ratio
            + self.consumer_num_ratio
            + self.controllable_consumer_num_ratio,
            1.0,
        )

        assert self.generator_spare >= 1.0
        assert self.controllable_consumer_spare >= 1.0

        assert self.initial_rebalance in ["directed", "undirected"]

    def from_dict(self, config: dict[str, Any]) -> None:
        # Pop distributions
        self.coupling_distribution = DistributionConfig(
            **config.pop("coupling_distribution")
        )
        self.generator_capacity_distribution = DistributionConfig(
            **config.pop("generator_capacity_distribution")
        )
        self.renewable_mass_distribution = DistributionConfig(
            **config.pop("renewable_mass_distribution")
        )
        self.renewable_capacity_distribution = DistributionConfig(
            **config.pop("renewable_capacity_distribution")
        )
        self.consumer_max_units_distribution = DistributionConfig(
            **config.pop("consumer_max_units_distribution")
        )
        self.controllable_consumer_capacity_distribution = DistributionConfig(
            **config.pop("controllable_consumer_capacity_distribution")
        )

        for key, value in config.items():
            setattr(self, key, value)


GRID_CONFIG = GridConfig()
