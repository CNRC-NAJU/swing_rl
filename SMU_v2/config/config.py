from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .grid import GRID_CONFIG, GridConfig
from .rl import RL_CONFIG, RLConfig


@dataclass(slots=True)
class Config:
    grid: GridConfig = field(default_factory=lambda: GRID_CONFIG)
    rl: RLConfig = field(default_factory=lambda: RL_CONFIG)

    @property
    def dict(self) -> dict[str, Any]:
        def remove_prefix_and_filter_none(data):
            return {k.strip("_"): v for k, v in dict(data).items() if v is not None}

        return asdict(self, dict_factory=remove_prefix_and_filter_none)

    def to_yaml(self, file_path: Path | str) -> None:
        with open(file_path, "w") as f:
            yaml.safe_dump(self.dict, f, sort_keys=False)

    def from_yaml(self, file_path: Path | str) -> None:
        with open(file_path, "r") as f:
            config: dict[str, Any] = yaml.safe_load(f)

        self.grid.from_dict(config.pop("grid"))
        self.rl.from_dict(config.pop("rl"))


CONFIG = Config()
