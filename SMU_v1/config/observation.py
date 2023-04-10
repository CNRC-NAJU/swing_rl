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

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            setattr(self, key, value)


OBSERVATION_CONFIG = ObservationConfig()
