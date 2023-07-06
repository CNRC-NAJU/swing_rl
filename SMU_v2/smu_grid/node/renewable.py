from __future__ import annotations

from config.grid import GRID_CONFIG, RenewableUnitConfig

from .node import Node
from .type import NodeType


class Renewable(Node):
    __slots__ = []

    def __init__(
        self, max_units: int, mass: float, config: RenewableUnitConfig | None = None
    ) -> None:
        super().__init__(max_units)

        if config is None:
            config = GRID_CONFIG.renewable

        self._unit_power = config.power
        self._unit_mass = mass
        self._unit_gamma = config.gamma_mass_ratio * mass

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
        return self._active_units * self._unit_mass

    @property
    def gamma(self) -> float:
        return self._active_units * self._unit_gamma
