from dataclasses import dataclass
from typing import Literal, get_args

DISTRIBUTION_NAME = Literal["uniform", "normal", "uniform_wo_avg", "normal_wo_avg"]


@dataclass(frozen=True, slots=True, kw_only=True)
class DistributionConfig:
    name: DISTRIBUTION_NAME = "uniform"  # Name of the distributions
    low: float | None = None  # uniform: lower bound
    high: float | None = None  # uniform: upper bound
    delta: float | None = None  # uniform_wo_avg: low=avg-delta, high=avg+delta
    avg: float | None = None  # normal: average
    std: float | None = None  # normal, normal_wo_avg: standard deviation

    def __post_init__(self) -> None:
        assert self.name in get_args(DISTRIBUTION_NAME)

        if self.name == "uniform":
            assert self.low is not None and self.high is not None
            assert self.low <= self.high

        elif self.name == "normal":
            assert self.avg is not None
            assert (self.std is not None) and (self.std >= 0)

        elif self.name == "uniform_wo_avg":
            assert self.delta is not None

        elif self.name == "normal_wo_avg":
            assert (self.std is not None) and (self.std >= 0)
