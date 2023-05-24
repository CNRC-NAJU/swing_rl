from dataclasses import dataclass
from typing import Any


@dataclass
class ObservationConfig:
    node_type: bool = True
    phase: bool = True
    dphase: bool = True
    mass: bool = True
    gamma: bool = True
    power: bool = True
    active_ratio: bool = True
    perturbation: bool = True
    edge_list: bool = True
    coupling: bool = True

    def __post_init__(self) -> None:
        assert self.phase, "You should observe phase"
        assert self.edge_list, "You should observe edge list"
        assert self.coupling, "You should observe coupling"

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)


OBSERVATION_CONFIG = ObservationConfig()
