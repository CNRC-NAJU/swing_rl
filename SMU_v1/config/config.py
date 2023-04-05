from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .consumer import ConsumerConfig
from .generator import GeneratorConfig
from .graph import GraphConfig
from .grid import GridConfig
from .observation import ObservationConfig
from .renewable import RenewableConfig
from .rl import RLConfig
from .singleton import Singleton
from .swing import SwingConfig


@dataclass
class Config(metaclass=Singleton):
    consumer: ConsumerConfig = ConsumerConfig()
    generator: GeneratorConfig = GeneratorConfig()
    renewable: RenewableConfig = RenewableConfig()

    graph: GraphConfig = GraphConfig()
    grid: GridConfig = GridConfig()

    observation: ObservationConfig = ObservationConfig()
    rl: RLConfig = RLConfig()
    swing: SwingConfig = SwingConfig()

    @property
    def dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, file_path: Path | str) -> None:
        with open(file_path, "w") as f:
            yaml.safe_dump(self.dict, f)

    @classmethod
    def from_yaml(cls, file_path: Path | str) -> Config:
        with open(file_path, "r") as f:
            config: dict[str, Any] = yaml.safe_load(f)

        ConsumerConfig.from_dict(config.pop("consumer"))
        GeneratorConfig.from_dict(config.pop("generator"))
        RenewableConfig.from_dict(config.pop("renewable"))

        GraphConfig.from_dict(config.pop("graph"))
        GridConfig.from_dict(config.pop("grid"))

        ObservationConfig.from_dict(config.pop("observation"))
        RLConfig.from_dict(config.pop("rl"))
        SwingConfig.from_dict(config.pop("swing"))

        return cls()

