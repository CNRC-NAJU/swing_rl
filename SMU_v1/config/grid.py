import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .distribution import DistributionConfig

_REBALANCE = Literal["directed", "undirected", "deterministic"]


@dataclass(slots=True)
class GridConfig:
    # Distribution of couplings of each nodes
    coupling_distribution: DistributionConfig = DistributionConfig(
        name="uniform", min=10.0, max=10.0
    )

    # Number ratio: generator + renewable + consumer + controllable consumer = 1.0
    _generator_num_ratio: float = 0.4
    _renewable_num_ratio: float = 0.2
    _consumer_num_ratio: float = 0.3
    _sink_num_ratio: float = 0.1

    # Distribution of nodes
    generator_capacity_distribution: DistributionConfig = DistributionConfig(
        name="uniform_wo_avg", delta=4.0
    )
    renewable_capacity_distribution: DistributionConfig = DistributionConfig(
        name="uniform_wo_avg", delta=4.0
    )
    renewable_mass_distribution: DistributionConfig = DistributionConfig(
        name="uniform", min=0.1, max=0.1
    )
    consumer_capacity_distribution: DistributionConfig = DistributionConfig(
        name="uniform", min=10.0, max=10.0
    )
    sink_capacity_distribution: DistributionConfig = DistributionConfig(
        name="uniform_wo_avg", delta=4.0
    )

    # Power capacity ratio
    _generator_spare: float = 1.1  # (generator capacity) = spare * (consumer capacity)
    _sink_spare: float = (
        1.1  # (controllable consumer capacity) = spare * (renewable capacity)
    )
    source_ratio: float = 4.0  # (renewable capacity) = (generator capacity) / ratio

    # Initial activeness
    _initial_active_ratio: float = 0.5
    _initial_rebalance: _REBALANCE = "directed"
    initial_max_rebalance: int = 1000

    def __post_init__(self) -> None:
        assert self.validate_num_ratio(
            self._generator_num_ratio,
            self._renewable_num_ratio,
            self._consumer_num_ratio,
            self._sink_num_ratio,
        )

        assert self.validate_spare(self._generator_spare)
        assert self.validate_spare(self._sink_spare)

        assert self.validate_active_ratio(self._initial_active_ratio)
        assert self.validate_initial_rebalance(self._initial_rebalance)

    def from_dict(self, config: dict[str, Any]) -> None:
        # Pop distributions
        self.coupling_distribution = DistributionConfig(
            **config.pop("coupling_distribution")
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
        self.consumer_capacity_distribution = DistributionConfig(
            **config.pop("consumer_capacity_distribution")
        )
        self.sink_capacity_distribution = DistributionConfig(
            **config.pop("sink_capacity_distribution")
        )

        num_ratios: dict[str, float] = {}
        for key, value in config.items():
            assert hasattr(self, key)
            if "num_ratio" in key:
                num_ratios[key.replace("_num_ratio", "")] = value
                continue
            setattr(self, key, value)
        self.set_num_ratio(**num_ratios)

    # --------------------- num_ratio -------------------------
    @staticmethod
    def validate_num_ratio(
        generator: float, renewable: float, consumer: float, sink: float
    ) -> bool:
        return np.isclose(1.0, generator + renewable + consumer + sink).item()

    @property
    def generator_num_ratio(self) -> float:
        return self._generator_num_ratio

    @property
    def renewable_num_ratio(self) -> float:
        return self._renewable_num_ratio

    @property
    def consumer_num_ratio(self) -> float:
        return self._consumer_num_ratio

    @property
    def sink_num_ratio(self) -> float:
        return self._sink_num_ratio

    @generator_num_ratio.setter
    def generator_num_ratio(self, _: float) -> None:
        warnings.warn(
            "You should not assign num_ratio one-by-one. Use set_num_ratio instead.",
            stacklevel=2,
        )

    @renewable_num_ratio.setter
    def renewable_num_ratio(self, _: float) -> None:
        warnings.warn(
            "You should not assign num_ratio one-by-one. Use set_num_ratio instead.",
            stacklevel=2,
        )

    @consumer_num_ratio.setter
    def consumer_num_ratio(self, _: float) -> None:
        warnings.warn(
            "You should not assign num_ratio one-by-one. Use set_num_ratio instead.",
            stacklevel=2,
        )

    @sink_num_ratio.setter
    def sink_num_ratio(self, _: float) -> None:
        warnings.warn(
            "You should not assign num_ratio one-by-one. Use set_num_ratio instead.",
            stacklevel=2,
        )

    def set_num_ratio(
        self, generator: float, renewable: float, consumer: float, sink: float
    ) -> None:
        if not self.validate_num_ratio(generator, renewable, consumer, sink):
            warnings.warn(f"Invalid num ratio. Ignore", stacklevel=2)
            return
        self._generator_num_ratio = generator
        self._renewable_num_ratio = renewable
        self._consumer_num_ratio = consumer
        self._sink_num_ratio = sink

    # --------------------- spare -------------------------
    @staticmethod
    def validate_spare(spare: float) -> bool:
        return spare >= 1.0

    @property
    def generator_spare(self) -> float:
        return self._generator_spare

    @generator_spare.setter
    def generator_spare(self, value: float) -> None:
        if not self.validate_spare(value):
            warnings.warn(
                "Generator spare should be large or equal to 1. Ignore", stacklevel=2
            )
            return
        self._generator_spare = value

    @property
    def sink_spare(self) -> float:
        return self._sink_spare

    @sink_spare.setter
    def sink_spare(self, value: float) -> None:
        if not self.validate_spare(value):
            warnings.warn(
                "Sink spare should be large or equal to 1. Ignore", stacklevel=2
            )
            return
        self._sink_spare = value

    # --------------------- Active ratio -------------------------
    @staticmethod
    def validate_active_ratio(ratio: float) -> bool:
        return 0.0 <= ratio <= 1.0

    @property
    def initial_active_ratio(self) -> float:
        return self._initial_active_ratio

    @initial_active_ratio.setter
    def initial_active_ratio(self, value: float) -> None:
        if not self.validate_active_ratio(value):
            warnings.warn("Invalid initial active ratio. Ignore", stacklevel=2)
            return
        self._initial_active_ratio = value

    # --------------------- Rebalance -------------------------
    @staticmethod
    def validate_initial_rebalance(rebalance: _REBALANCE) -> bool:
        if rebalance == "undirected":
            warnings.warn(
                "Undirected rebalancing strategy may endup Error", stacklevel=2
            )
        return rebalance in ["directed", "undirected", "deterministic"]

    @property
    def initial_rebalance(self) -> _REBALANCE:
        return self._initial_rebalance

    @initial_rebalance.setter
    def initial_rebalance(self, value: _REBALANCE) -> None:
        if not self.validate_initial_rebalance(value):
            warnings.warn("Invalid initial rebalance strategy. Ignore", stacklevel=2)
            return
        self._initial_rebalance = value


GRID_CONFIG = GridConfig()
