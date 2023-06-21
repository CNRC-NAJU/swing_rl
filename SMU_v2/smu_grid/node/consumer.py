from __future__ import annotations

from config.grid import GRID_CONFIG, UnitConfig

from .node import Node
from .type import NodeType


class Consumer(Node):
    __slots__ = []

    def __init__(self, max_units: int, config: UnitConfig | None = None) -> None:
        super().__init__(max_units)

        if config is None:
            config = GRID_CONFIG.consumer

        self._unit_power = config.power
        self._unit_mass = config.mass
        self._unit_gamma = config.gamma

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
