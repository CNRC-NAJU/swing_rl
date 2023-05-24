from dataclasses import dataclass
from typing import Any


## When finding steady state, decrease mass, increase gamma
@dataclass
class NodeConfig:
    # ---------------- Generator ----------------
    generator_unit_power: int = 1  # This should not be changed
    generator_unit_mass: float = 1.0
    generator_unit_gamma: float = 1.0
    # generator_temporary_unit_mass: float = 0.1
    # generator_temporary_unit_gamma: float = 10.0

    # ---------------- Renewable ----------------
    renewable_unit_power: int = 1  # This should not be changed
    renewable_unit_gamma_mass_ratio: float = 1.0  # gamma / mass

    # ---------------- Consumer ----------------
    consumer_unit_power: int = -1  # This should not be changed
    consumer_unit_mass: float = 1.0
    consumer_unit_gamma: float = 1.0

    # -------- Controllable Consumer ---------
    sink_unit_power: int = -1  # This should not be changed
    sink_unit_mass: float = 1.0
    sink_unit_gamma: float = 1.0

    def __post__init__(self) -> None:
        assert self.generator_unit_power == 1
        assert self.renewable_unit_power == 1
        assert self.consumer_unit_power == -1
        assert self.sink_unit_power == -1

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)


NODE_CONFIG = NodeConfig()
