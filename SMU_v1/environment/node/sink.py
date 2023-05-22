from __future__ import annotations

from config import NODE_CONFIG

from .node import Node
from .type import NodeType


class Sink(Node):
    def __init__(self, max_units: int) -> None:
        super().__init__(max_units)

        self.unit_power = NODE_CONFIG.sink_unit_power
        self.unit_mass = NODE_CONFIG.sink_unit_mass
        self.unit_gamma = NODE_CONFIG.sink_unit_gamma

    @classmethod
    def from_capacity(cls, capacity: int) -> Sink:
        assert capacity >= 0, "Capacity should be always positive"
        return cls(capacity)

    @property
    def type(self) -> NodeType:
        return NodeType.SINK

    @property
    def power(self) -> int:
        """Negative value: power consumption"""
        return self.active_units * self.unit_power

    @property
    def mass(self) -> float:
        return self.active_units * self.unit_mass

    @property
    def gamma(self) -> float:
        return self.active_units * self.unit_gamma

    @property
    def capacity(self) -> int:
        return self.max_units * self.unit_power
