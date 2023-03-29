from __future__ import annotations

from config import GeneratorConfig

from .node import Node


class Generator(Node):
    def __init__(self, max_units: int = 0) -> None:
        super().__init__(max_units)
        self.active_units = 0  # [-max_units, max_units]

        config = GeneratorConfig()
        self.unit_power = config.unit_power
        self.unit_mass = config.unit_mass
        self.unit_gamma = config.unit_gamma

    @classmethod
    def from_capacity(cls, capacity: int) -> Generator:
        config = GeneratorConfig()
        assert capacity % config.unit_power == 0, f"Capacity not valid: {capacity}"

        return cls(capacity // config.unit_power)

    def __str__(self) -> str:
        return "generator"

    @property
    def full_inactive(self) -> bool:
        return self.active_units == -self.max_units

    @property
    def power(self) -> int:
        # Could be negative
        return self.active_units * self.unit_power

    @property
    def mass(self) -> float:
        return abs(self.active_units) * self.unit_mass

    @property
    def gamma(self) -> float:
        return abs(self.active_units) * self.unit_gamma

    @property
    def capacity(self) -> int:
        return self.max_units * self.unit_power
