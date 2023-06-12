from __future__ import annotations

from config.grid import GRID_CONFIG

from .node import Node
from .type import NodeType


class Consumer(Node):
    __slots__ = []

    def __init__(
        self,
        max_units: int,
        unit_power: int = GRID_CONFIG.consumer.power,
        unit_mass: float = GRID_CONFIG.consumer.mass,
        unit_gamma: float = GRID_CONFIG.consumer.gamma,
    ) -> None:
        super().__init__(max_units)

        self._unit_power = unit_power
        self._unit_mass = unit_mass
        self._unit_gamma = unit_gamma

    @classmethod
    def from_capacity(cls, capacity: int) -> Consumer:
        return cls(capacity)

    @property
    def type(self) -> NodeType:
        return NodeType.CONSUMER

    @property
    def power(self) -> int:
        """Negative value: power consumption"""
        return self._active_units * self._unit_power

    @property
    def mass(self) -> float:
        return self._active_units * self._unit_mass

    @property
    def gamma(self) -> float:
        return self._active_units * self._unit_gamma