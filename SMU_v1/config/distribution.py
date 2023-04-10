from dataclasses import dataclass


@dataclass
class DistributionConfig:
    name: str = "uniform"
    min: float | None = None
    max: float | None = None
    delta: float | None = None
    avg: float | None = None
    std: float | None = None

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
