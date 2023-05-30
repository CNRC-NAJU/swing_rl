from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ObservationConfig:
    node_type: bool = True
    _phase: bool = True
    dphase: bool = True
    mass: bool = True
    gamma: bool = True
    power: bool = True
    active_ratio: bool = True
    perturbation: bool = True
    _edge_list: bool = True
    _coupling: bool = True

    def __post_init__(self) -> None:
        assert self._phase
        assert self._edge_list
        assert self._coupling

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    @property
    def phase(self) -> bool:
        return self._phase

    @phase.setter
    def phase(self, value: bool) -> None:
        if not value:
            print("You should always observe phase. Ignore")

    @property
    def edge_list(self) -> bool:
        return self._edge_list

    @edge_list.setter
    def edge_list(self, value: bool) -> None:
        if not value:
            print("You should always observe edge list. Ignore")

    @property
    def coupling(self) -> bool:
        return self._coupling

    @coupling.setter
    def coupling(self, value: bool) -> None:
        if not value:
            print("You should always observe coupling constant. Ignore")


OBSERVATION_CONFIG = ObservationConfig()
