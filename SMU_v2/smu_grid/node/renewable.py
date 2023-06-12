from __future__ import annotations

from config.grid import GRID_CONFIG

from .node import Node
from .type import NodeType


class Renewable(Node):
    __slots__ = []

    def __init__(
        self,
        max_units: int,
        mass: float,
        unit_power: int = GRID_CONFIG.renewable.power,
        gamma_mass_ratio: float = GRID_CONFIG.renewable.gamma_mass_ratio,
    ) -> None:
        super().__init__(max_units)

        self._unit_power = unit_power
        self._unit_mass = mass
        self._unit_gamma = gamma_mass_ratio * mass

    @classmethod
    def from_capacity(cls, capacity: int, mass: float) -> Renewable:
        return cls(capacity, mass)

    @property
    def type(self) -> NodeType:
        return NodeType.RENEWABLE

    @property
    def power(self) -> int:
        return self._active_units * self._unit_power

    @property
    def mass(self) -> float:
        return self._unit_mass

    @property
    def gamma(self) -> float:
        return self._unit_gamma
