from __future__ import annotations

from typing import cast

import numpy as np
from config import ConsumerConfig

from .node import Node
from .type import NodeType


class Consumer(Node):
    def __init__(self, max_units: int = 0) -> None:
        super().__init__(max_units)
        self.active_units = 0  # [0, max_units]

        self.unit_power = ConsumerConfig.unit_power
        self.unit_mass = ConsumerConfig.unit_mass
        self.unit_gamma = ConsumerConfig.unit_gamma

    @classmethod
    def randomly(cls, rng: np.random.Generator) -> Consumer:
        """
        Randomly create consumer, with max_units following certain distribution,
        specified in configuration
        """
        max_units_distribution = ConsumerConfig.max_units_distribution
        if max_units_distribution.name == "uniform":
            max_units = rng.integers(
                low=int(cast(float, max_units_distribution.min)),
                high=int(cast(float, max_units_distribution.max)),
                endpoint=True,
            )
        elif max_units_distribution.name == "normal":
            max_units = rng.normal(
                loc=cast(float, max_units_distribution.avg),
                scale=cast(float, max_units_distribution.std),
            )
            max_units = max(1, round(max_units))  # Clipping: max_units > 1
        else:
            raise ValueError(f"No such distribution {max_units_distribution.name}")

        return cls(max_units)

    @property
    def type(self) -> NodeType:
        return NodeType.CONSUMER

    @property
    def power(self) -> int:
        return self.active_units * self.unit_power

    @property
    def mass(self) -> float:
        if self.active_units == 0:
            return self.unit_mass
        return self.active_units * self.unit_mass

    @property
    def gamma(self) -> float:
        if self.active_units == 0:
            return self.unit_gamma
        return self.active_units * self.unit_gamma

    @property
    def capacity(self) -> int:
        return self.max_units * self.unit_power
