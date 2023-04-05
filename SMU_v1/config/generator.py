from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .singleton import Singleton


@dataclass
class GeneratorConfig(metaclass=Singleton):
    # Unit
    unit_power: int = 1
    unit_mass: float = 1.0
    unit_gamma: float = 1.0

    # Distribution of capacity
    capacity_distribution_name: str = "uniform"
    capacity_distribution_param: float = 4.0  # Delta for uniform/sigma for normal

    def __post__init__(self) -> None:
        assert self.capacity_distribution_name in ["uniform", "normal"]

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GeneratorConfig:
        generator = cls()
        for key, value in config.items():
            setattr(generator, key, value)

        return generator
