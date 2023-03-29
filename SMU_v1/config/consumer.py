from dataclasses import dataclass

from .distribution import DistributionConfig


@dataclass
class ConsumerConfig:
    # Unit
    unit_power: int = -1
    unit_mass: float = 1.0
    unit_gamma: float = 1.0

    # Distribution of max units
    max_units_distribution: DistributionConfig = DistributionConfig(
        name="uniform", min=10.0, max=10.0
    )
