from __future__ import annotations

from config import GeneratorConfig

from .node import Node
from .type import NodeType


class Generator(Node):
    def __init__(self, max_units: int = 0) -> None:
        super().__init__(max_units)

        config = GeneratorConfig()
        self.unit_power = config.unit_power
        self.unit_mass = config.unit_mass
        self.unit_gamma = config.unit_gamma

    @classmethod
    def from_capacity(cls, capacity: int) -> Generator:
        config = GeneratorConfig()
        assert capacity % config.unit_power == 0, f"Capacity not valid: {capacity}"

        return cls(capacity // config.unit_power)

    @property
    def type(self) -> NodeType:
        return NodeType.GENERATOR

    @property
    def power(self) -> int:
        return self.active_units * self.unit_power

    @property
    def mass(self) -> float:
        if self.active_units == 0:
            return self.unit_mass
        return abs(self.active_units) * self.unit_mass

    @property
    def gamma(self) -> float:
        if self.active_units == 0:
            return self.unit_gamma
        return abs(self.active_units) * self.unit_gamma

    @property
    def capacity(self) -> int:
        return self.max_units * self.unit_power
