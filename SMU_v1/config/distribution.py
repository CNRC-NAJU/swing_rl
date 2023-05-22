from dataclasses import dataclass
from typing import Literal


@dataclass
class DistributionConfig:
    name: Literal["uniform", "normal", "uniform_wo_avg", "normal_wo_avg"] = (
        "uniform"  # Name of the distributions
    )
    min: float | None = None  # uniform: minimum
    max: float | None = None  # uniform: maximum
    delta: float | None = None  # uniform_wo_avg: min=avg-delta, max=avg+delta
    avg: float | None = None  # normal: average
    std: float | None = None  # normal, normal_wo_avg: standard deviation

    def __post_init__(self) -> None:
        if self.name == "uniform":
            assert self.min is not None
            assert self.max is not None
        elif self.name == "normal":
            assert self.avg is not None
            assert self.std is not None
        elif self.name == "uniform_wo_avg":
            assert self.delta is not None
        elif self.name == "normal_wo_avg":
            assert self.std is not None
        else:
            raise ValueError(f"No such distribution name: {self.name}")
