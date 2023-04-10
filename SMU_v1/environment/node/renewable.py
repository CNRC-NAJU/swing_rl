from __future__ import annotations

from typing import cast

import numpy as np
from config import NODE_CONFIG

from .node import Node
from .type import NodeType

Rng = np.random.Generator | int | None


class Renewable(Node):
    def __init__(self, max_units: int = 0, mass: float = 0.0) -> None:
        super().__init__(max_units)

        self.unit_power = NODE_CONFIG.renewable_unit_power
        self.unit_mass = mass
        self.unit_gamma = NODE_CONFIG.renewable_unit_gamma_mass_ratio * mass

    @classmethod
    def from_capacity(cls, capacity: int, mass: float) -> Renewable:
        assert (
            capacity % NODE_CONFIG.renewable_unit_power == 0
        ), f"Capacity not valid: {capacity}"
        return cls(capacity // NODE_CONFIG.renewable_unit_power, mass)

    # def randomly_from_capacity(cls, capacity: int, rng: Rng) -> Renewable:
    #     """
    #     Randomly create renewable, with given capacity
    #     """
    #     assert capacity % NODE_CONFIG.renewable_unit_power == 0, f"Capacity not valid: {capacity}"

    #     if not isinstance(rng, np.random.Generator):
    #         rng = np.random.default_rng(rng)
    #     mass_distribution = NODE_CONFIG.renewable_mass_distribution
    #     if mass_distribution.name == "uniform":
    #         mass = rng.uniform(
    #             low=cast(float, mass_distribution.min),
    #             high=cast(float, mass_distribution.max),
    #         )
    #     elif mass_distribution.name == "normal":
    #         mass = rng.normal(
    #             loc=cast(float, mass_distribution.avg),
    #             scale=cast(float, mass_distribution.std),
    #         )
    #         mass = max(0.0, round(mass))  # clipping: mass >= 0
    #     else:
    #         raise NotImplementedError(f"No such distribution {mass_distribution.name}")
    #     return cls(capacity // NODE_CONFIG.unit_power, mass)

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
