from __future__ import annotations

from config import NODE_CONFIG

from .node import Node
from .type import NodeType


class Renewable(Node):
    def __init__(self, max_units: int, mass: float) -> None:
        super().__init__(max_units)

        self.unit_power = NODE_CONFIG.renewable_unit_power
        self.unit_mass = mass
        self.unit_gamma = NODE_CONFIG.renewable_unit_gamma_mass_ratio * mass

    @classmethod
    def from_capacity(cls, capacity: int, mass: float) -> Renewable:
        assert capacity >= 0, "Capacity should be positive"
        assert (
            capacity % NODE_CONFIG.renewable_unit_power == 0
        ), f"Capacity not valid: {capacity}"
        return cls(capacity // NODE_CONFIG.renewable_unit_power, mass)

    @property
    def type(self) -> NodeType:
        return NodeType.RENEWABLE

    @property
    def power(self) -> int:
        return self.active_units * self.unit_power

    @property
    def mass(self) -> float:
        return self.unit_mass

    @property
    def gamma(self) -> float:
        return self.unit_gamma

    @property
    def capacity(self) -> int:
        return self.max_units * self.unit_power
