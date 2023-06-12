from dataclasses import dataclass
from typing import Any, Literal
import warnings

OBSERVATION = Literal[
    "node_type",
    "phase",
    "dphase",
    "mass",
    "gamma",
    "power",
    "active_ratio",
    "perturbation",
    "edge_list",
    "coupling",
]


@dataclass(slots=True)
class ObservationConfig:
    node_type: bool = True
    phase: bool = True
    dphase: bool = True
    mass: bool = True
    gamma: bool = True
    power: bool = True
    active_ratio: bool = True
    _perturbation: bool = True
    _edge_list: bool = True
    _coupling: bool = True

    def __post_init__(self) -> None:
        assert self._perturbation
        assert self._edge_list
        assert self._coupling

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            if key in ["perturbation", "edge_list", "coupling"]:
                continue
            setattr(self, key, value)

    @property
    def perturbation(self) -> bool:
        return self._perturbation

    @perturbation.setter
    def perturbation(self, _: bool) -> None:
        warnings.warn("You should observe perturbation. sIgnore", stacklevel=2)
        return

    @property
    def edge_list(self) -> bool:
        return self._edge_list

    @edge_list.setter
    def edge_list(self, _: bool) -> None:
        warnings.warn("You should observe edge list. Ignore", stacklevel=2)
        return

    @property
    def coupling(self) -> bool:
        return self._coupling

    @coupling.setter
    def coupling(self, _: bool) -> None:
        warnings.warn("You should observe coupling constant. Ignore", stacklevel=2)
        return
