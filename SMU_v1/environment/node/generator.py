from __future__ import annotations

from config import NODE_CONFIG

from .node import Node
from .type import NodeType


class Generator(Node):
    def __init__(self, max_units: int) -> None:
        super().__init__(max_units)

        self.unit_power = NODE_CONFIG.generator_unit_power
        self.unit_mass = NODE_CONFIG.generator_unit_mass
        self.unit_gamma = NODE_CONFIG.generator_unit_gamma

    @classmethod
    def from_capacity(cls, capacity: int) -> Generator:
        assert (
            capacity % NODE_CONFIG.generator_unit_power == 0
        ), f"Capacity not valid: {capacity}"

        return cls(capacity // NODE_CONFIG.generator_unit_power)

    @property
    def type(self) -> NodeType:
        return NodeType.GENERATOR

    @property
    def power(self) -> int:
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
