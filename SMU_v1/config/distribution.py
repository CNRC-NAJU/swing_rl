from dataclasses import dataclass
from typing import Literal, get_args

_DISTRIBUTION_NAME = Literal["uniform", "normal", "uniform_wo_avg", "normal_wo_avg"]


@dataclass(frozen=True, slots=True, kw_only=True)
class DistributionConfig:
    name: _DISTRIBUTION_NAME = "uniform"  # Name of the distributions
    min: float | None = None  # uniform: minimum
    max: float | None = None  # uniform: maximum
    delta: float | None = None  # uniform_wo_avg: min=avg-delta, max=avg+delta
    avg: float | None = None  # normal: average
    std: float | None = None  # normal, normal_wo_avg: standard deviation

    def __post_init__(self) -> None:
        assert self.name in get_args(_DISTRIBUTION_NAME)

        if self.name == "uniform":
            assert (self.min is not None) and (self.max is not None)
        elif self.name == "normal":
            assert (self.avg is not None) and (self.std is not None)
        elif self.name == "uniform_wo_avg":
            assert self.delta is not None
        elif self.name == "normal_wo_avg":
            assert self.std is not None
