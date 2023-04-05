from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .distribution import DistributionConfig
from .singleton import Singleton


@dataclass
class ConsumerConfig(metaclass=Singleton):
    # Unit
    unit_power: int = -1
    unit_mass: float = 1.0
    unit_gamma: float = 1.0

    # Distribution of max units
    max_units_distribution: DistributionConfig = DistributionConfig(
        name="uniform", min=10.0, max=10.0
    )


    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ConsumerConfig:
        distribution = DistributionConfig(**config.pop("max_units_distribution"))

        consumer = cls(max_units_distribution=distribution)
        for key, value in config.items():
            setattr(consumer, key, value)

        return consumer