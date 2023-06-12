import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from config.distribution import DistributionConfig

from .graph import GraphConfig
from .monitor import MonitorConfig
from .perturbation import PerturbationConfig
from .swing import SwingConfig
from .turn_on import TurnOnConfig
from .unit import RenewableUnitConfig, UnitConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class NumRatioConfig:
    """Number ratio of each node types"""

    consumer: float = 0.3
    generator: float = 0.4
    renewable: float = 0.2
    sink: float = 0.1

    def __post_init__(self) -> None:
        assert self.validate_num_ratio(
            self.consumer, self.generator, self.renewable, self.sink
        )

    @staticmethod
    def validate_num_ratio(
        consumer: float, generator: float, renewable: float, sink: float
    ) -> bool:
        return np.isclose(1.0, generator + renewable + consumer + sink).item()


@dataclass(slots=True)
class GridConfig:
    # Graph configuration
    graph: GraphConfig = GraphConfig()

    # Coupling constants configuration
    _coupling_distribution: DistributionConfig = DistributionConfig(
        name="uniform", low=10.0, high=10.0
    )

    # Node types configuration
    num_ratio: NumRatioConfig = NumRatioConfig()

    # Consumer configuration
    consumer: UnitConfig = UnitConfig(_power=-1, mass=1.0, gamma=1.0)
    _consumer_capacity_distribution: DistributionConfig = DistributionConfig(
        name="uniform", low=10.0, high=10.0
    )

    # Generator configuration
    generator: UnitConfig = UnitConfig(_power=1, mass=1.0, gamma=1.0)
    _generator_spare: float = 1.1  # (generator capacity) = spare * (consumer capacity)
    _generator_capacity_distribution: DistributionConfig = DistributionConfig(
        name="uniform_wo_avg", low=2.0, delta=4.0
    )

    # Renewable configuration
    renewable: RenewableUnitConfig = RenewableUnitConfig(_power=1, gamma_mass_ratio=1.0)
    source_ratio: float = 0.25  # (renewable capacity) = ratio * (generator capacity)
    _renewable_capacity_distribution: DistributionConfig = DistributionConfig(
        name="uniform_wo_avg", low=2.0, delta=4.0
    )
    _renewable_mass_distribution: DistributionConfig = DistributionConfig(
        name="uniform", low=0.1, high=0.1
    )

    # Sink configuration
    sink: UnitConfig = UnitConfig(_power=-1, mass=1.0, gamma=1.0)
    _sink_spare: float = 1.1  # (sink capacity) = spare * (renewable capacity)
    _sink_capacity_distribution: DistributionConfig = DistributionConfig(
        name="uniform_wo_avg", low=2.0, delta=4.0
    )

    # Turn on, perturbation, Steady state
    turn_on: TurnOnConfig = TurnOnConfig()
    perturbation: PerturbationConfig = PerturbationConfig()
    steady: SwingConfig = SwingConfig(
        _name="rk4",
        dt=1e-2,
        max_time=40.0,
        monitor=MonitorConfig("inside", 1e-3)
    )

    def __post_init__(self) -> None:
        # Consumer
        assert self.validate_capacity_distribution(
            self._consumer_capacity_distribution, without_average=False
        )

        # Generator
        assert self.validate_spare(self._generator_spare)
        assert self.validate_capacity_distribution(
            self._generator_capacity_distribution, without_average=True
        )

        # Renewable
        assert self.validate_capacity_distribution(
            self._renewable_capacity_distribution, without_average=True
        )
        assert self.validate_mass_distribution(self._renewable_mass_distribution)

        # Sink
        assert self.validate_spare(self._sink_spare)
        assert self.validate_capacity_distribution(
            self._sink_capacity_distribution, without_average=True
        )

    def from_dict(self, config: dict[str, Any]) -> None:
        # Pop sub configurations
        self.graph.from_dict(config.pop("graph"))
        self.num_ratio = NumRatioConfig(**config.pop("num_ratio"))

        self.consumer.from_dict(config.pop("consumer"))
        self.generator.from_dict(config.pop("generator"))
        self.renewable.from_dict(config.pop("renewable"))
        self.sink.from_dict(config.pop("sink"))

        self.turn_on.from_dict(config.pop("turn_on"))
        self.perturbation.from_dict(config.pop("perturbation"))
        self.steady.from_dict(config.pop("steady"))

        # Pop distributions
        self.consumer_capacity_distribution = DistributionConfig(
            **config.pop("consumer_capacity_distribution")
        )
        self.generator_capacity_distribution = DistributionConfig(
            **config.pop("generator_capacity_distribution")
        )
        self.renewable_mass_distribution = DistributionConfig(
            **config.pop("renewable_mass_distribution")
        )
        self.renewable_capacity_distribution = DistributionConfig(
            **config.pop("renewable_capacity_distribution")
        )
        self.sink_capacity_distribution = DistributionConfig(
            **config.pop("sink_capacity_distribution")
        )

        self.coupling_distribution = DistributionConfig(
            **config.pop("coupling_distribution")
        )

        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    # --------------------- Capacity distribution -------------------------
    @staticmethod
    def validate_capacity_distribution(
        distribution: DistributionConfig, without_average: bool
    ) -> bool:
        valid_min = distribution.low is not None and distribution.low >= 2.0
        if without_average:
            valid_name = "wo_avg" in distribution.name
        else:
            valid_name = "wo_avg" not in distribution.name

        return valid_min and valid_name

    @property
    def consumer_capacity_distribution(self) -> DistributionConfig:
        return self._consumer_capacity_distribution

    @consumer_capacity_distribution.setter
    def consumer_capacity_distribution(self, distribution: DistributionConfig) -> None:
        if not self.validate_capacity_distribution(distribution, without_average=False):
            warnings.warn("Invalid capacity distribution. Ignore", stacklevel=2)
            return
        self._consumer_capacity_distribution = distribution

    @property
    def generator_capacity_distribution(self) -> DistributionConfig:
        return self._generator_capacity_distribution

    @generator_capacity_distribution.setter
    def generator_capacity_distribution(self, distribution: DistributionConfig) -> None:
        if not self.validate_capacity_distribution(distribution, without_average=True):
            warnings.warn("Invalid capacity distribution. Ignore", stacklevel=2)
            return
        self._generator_capacity_distribution = distribution

    @property
    def renewable_capacity_distribution(self) -> DistributionConfig:
        return self._renewable_capacity_distribution

    @renewable_capacity_distribution.setter
    def renewable_capacity_distribution(self, distribution: DistributionConfig) -> None:
        if not self.validate_capacity_distribution(distribution, without_average=True):
            warnings.warn("Invalid capacity distribution. Ignore", stacklevel=2)
            return
        self._renewable_capacity_distribution = distribution

    @property
    def sink_capacity_distribution(self) -> DistributionConfig:
        return self._sink_capacity_distribution

    @sink_capacity_distribution.setter
    def sink_capacity_distribution(self, distribution: DistributionConfig) -> None:
        if not self.validate_capacity_distribution(distribution, without_average=True):
            warnings.warn("Invalid capacity distribution. Ignore", stacklevel=2)
            return
        self._sink_capacity_distribution = distribution

    # --------------------- other distribution -------------------------
    @staticmethod
    def validate_mass_distribution(distribution: DistributionConfig) -> bool:
        valid_min = distribution.low is not None and distribution.low > 0.0
        valid_name = "wo_avg" not in distribution.name
        return valid_min and valid_name

    @property
    def renewable_mass_distribution(self) -> DistributionConfig:
        return self._renewable_mass_distribution

    @renewable_mass_distribution.setter
    def renewable_mass_distribution(self, distribution: DistributionConfig) -> None:
        if not self.validate_mass_distribution(distribution):
            warnings.warn("Invalid mass distribution. Ignore", stacklevel=2)
            return
        self._renewable_capacity_distribution = distribution

    @staticmethod
    def validate_coupling_distribution(distribution: DistributionConfig) -> bool:
        return distribution.name in ["uniform", "normal"]

    @property
    def coupling_distribution(self) -> DistributionConfig:
        return self._coupling_distribution

    @coupling_distribution.setter
    def coupling_distribution(self, distribution: DistributionConfig) -> None:
        if not self.validate_coupling_distribution(distribution):
            warnings.warn(
                "Invalid coupling constant distribution. Ignore", stacklevel=2
            )
            return
        self._coupling_distribution = distribution

    # --------------------- spare -------------------------
    @staticmethod
    def validate_spare(spare: float) -> bool:
        return spare >= 1.0

    @property
    def generator_spare(self) -> float:
        return self._generator_spare

    @generator_spare.setter
    def generator_spare(self, spare: float) -> None:
        if not self.validate_spare(spare):
            warnings.warn(
                "Generator spare should be large or equal to 1. Ignore", stacklevel=2
            )
            return
        self._generator_spare = spare

    @property
    def sink_spare(self) -> float:
        return self._sink_spare

    @sink_spare.setter
    def sink_spare(self, spare: float) -> None:
        if not self.validate_spare(spare):
            warnings.warn(
                "Sink spare should be large or equal to 1. Ignore", stacklevel=2
            )
            return
        self._sink_spare = spare


GRID_CONFIG = GridConfig()
