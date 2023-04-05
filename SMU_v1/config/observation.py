from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .singleton import Singleton


@dataclass
class ObservationConfig(metaclass=Singleton):
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

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ObservationConfig:
        observation = cls()
        for key, value in config.items():
            setattr(observation, key, value)

        return observation
