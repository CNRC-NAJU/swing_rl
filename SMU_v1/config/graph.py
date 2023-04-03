from dataclasses import dataclass

from .distribution import DistributionConfig


@dataclass
class GraphConfig:
    topology: str = "shk"

    # Distribution of network size
    num_nodes_distribution: DistributionConfig = DistributionConfig(
        name="uniform", min=20.0, max=20.0
    )

    mean_degree: float = 4.0

    # shk parameters
    shk_p: float = 0.2
    shk_q: float = 0.3
    shk_r: float = 0.3333
    shk_s: float = 0.1
    shk_initial: int = 1


    def __post__init__(self) -> None:
        assert self.topology in ["shk", "ba", "er", "rr"]
