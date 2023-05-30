from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .agent import AGENT_CONFIG, AgentConfig
from .graph import GRAPH_CONFIG, GraphConfig
from .grid import GRID_CONFIG, GridConfig
from .node import NODE_CONFIG, NodeConfig
from .observation import OBSERVATION_CONFIG, ObservationConfig
from .rl import RL_CONFIG, RLConfig
from .swing import SWING_CONFIG, SwingConfig


@dataclass(slots=True)
class Config:
    node: NodeConfig = NODE_CONFIG
    graph: GraphConfig = GRAPH_CONFIG
    grid: GridConfig = GRID_CONFIG

    observation: ObservationConfig = OBSERVATION_CONFIG
    rl: RLConfig = RL_CONFIG
    swing: SwingConfig = SWING_CONFIG
    agent: AgentConfig = AGENT_CONFIG

    @property
    def dict(self) -> dict[str, Any]:
        def remove_prefix(data):
            return {k.strip("_"): v for k, v in dict(data).items()}

        return asdict(self, dict_factory=remove_prefix)

    def to_yaml(self, file_path: Path | str) -> None:
        with open(file_path, "w") as f:
            yaml.safe_dump(self.dict, f)

    def from_yaml(self, file_path: Path | str) -> None:
        with open(file_path, "r") as f:
            config: dict[str, Any] = yaml.safe_load(f)

        self.node.from_dict(config.pop("node"))
        self.graph.from_dict(config.pop("graph"))
        self.grid.from_dict(config.pop("grid"))

        self.observation.from_dict(config.pop("observation"))
        self.rl.from_dict(config.pop("rl"))
        self.swing.from_dict(config.pop("swing"))
        self.agent.from_dict(config.pop("agent"))


CONFIG = Config()
