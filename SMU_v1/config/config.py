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
from .swing import SwingConfig


@dataclass
class Config:
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

        consumer = ConsumerConfig(**config.pop("consumer"))
        generator = GeneratorConfig(**config.pop("generator"))
        renewable = RenewableConfig(**config.pop("renewable"))

        graph = GraphConfig(**config.pop("graph"))
        grid = GridConfig(**config.pop("grid"))
        observation = ObservationConfig(**config.pop("observation"))
        rl = RLConfig(**config.pop("rl"))
        swing = SwingConfig(**config.pop("swing"))

        return cls(
            **config,
            consumer=consumer,
            generator=generator,
            renewable=renewable,
            graph=graph,
            grid=grid,
            observation=observation,
            rl=rl,
            swing=swing,
        )

