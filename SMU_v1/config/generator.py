from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    # Unit
    unit_power: int = 1
    unit_mass: float = 1.0
    unit_gamma: float = 1.0

    # Distribution of capacity
    capacity_distribution_name: str = "uniform"
    capacity_distribution_param: float = 4.0  # Delta for uniform/sigma for normal

    def __post__init__(self) -> None:
        assert self.capacity_distribution_name in ["uniform", "normal"]
