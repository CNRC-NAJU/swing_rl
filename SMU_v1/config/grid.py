from dataclasses import dataclass

from .distribution import DistributionConfig


@dataclass
class GridConfig:
    # Distribution of couplings of each nodes
    coupling_distribution: DistributionConfig = DistributionConfig(
        name="uniform",
        min=1.0,
        max=1.0,
    )

    # Number ratio: generator + renewable + consumer = 1.0
    generator_num_ratio: float = 0.1
    renewable_num_ratio: float = 0.1
    consumer_num_ratio: float = 0.8

    # Power capacity ratio
    generator_spare: float = 1.1  # (generator capacity) = spare * (consumer capacity)
    source_ratio: float = 4.0  # (renewable capacity) = (generator capacity) / ratio

    # Initial activeness
    initial_active_ratio: float = 0.5
    initial_rebalance: str = "directed"

    def __post_init__(self) -> None:
        assert (
            self.generator_num_ratio
            + self.renewable_num_ratio
            + self.consumer_num_ratio
            == 1.0
        )

        assert self.initial_rebalance in ["directed", "undirected"]
