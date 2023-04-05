from dataclasses import dataclass

from .distribution import DistributionConfig


@dataclass
class RenewableConfig:
    # Unit
    unit_power: int = 1
    unit_gamma_mass_ratio: float = 1.0  # gamma / mass

    # Distribution of unit mass
    mass_distribution: DistributionConfig = DistributionConfig(
        name="uniform", min=0.1, max=0.1
    )

    # Distribution of capacity
    capacity_distribution_name: str = "uniform"
    capacity_distribution_param: float = 4.0  # Delta for uniform/sigma for normal

    def __post__init__(self) -> None:
        assert self.capacity_distribution_name in ["uniform", "normal"]
