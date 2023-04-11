from dataclasses import dataclass
from typing import Any


@dataclass
class NodeConfig:
    # ---------------- Generator ----------------
    generator_unit_power: int = 1
    generator_unit_mass: float = 1.0
    generator_unit_gamma: float = 1.0

    # ---------------- Renewable ----------------
    renewable_unit_power: int = 1
    renewable_unit_gamma_mass_ratio: float = 1.0  # gamma / mass

    # ---------------- Consumer ----------------
    consumer_unit_power: int = -1
    consumer_unit_mass: float = 1.0
    consumer_unit_gamma: float = 1.0

    # -------- Controllable Consumer ---------
    controllable_consumer_unit_power: int = -1
    controllable_consumer_unit_mass: float = 1.0
    controllable_consumer_unit_gamma: float = 1.0

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            setattr(self, key, value)


NODE_CONFIG = NodeConfig()
