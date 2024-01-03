from dataclasses import dataclass, field
from typing import Any

from config.grid import MonitorConfig, RebalanceConfig, ResetConfig, SwingConfig

from .agent import AgentConfig
from .observation import ObservationConfig
from .reward import RewardConfig


@dataclass(slots=True)
class RLConfig:
    # Trajectory for checking failed nodes, calculating reward
    swing: SwingConfig = field(
        default_factory=lambda: SwingConfig(
            _name="rk4", dt=1e-2, max_time=2.0, monitor=MonitorConfig("outside", 1.0)
        )
    )

    # Reward functions
    reward: RewardConfig = field(
        default_factory=lambda: RewardConfig(_name="weighted_area", threshold=0.0)
    )
    reward_failed: RewardConfig = field(
        default_factory=lambda: RewardConfig(_name="inverse_time", scale=10.0)
    )
    reward_failed_rebalance: RewardConfig = field(
        default_factory=lambda: RewardConfig(_name="constant", scale=100.0)
    )
    reward_failed_steady: RewardConfig = field(
        default_factory=lambda: RewardConfig(_name="constant", scale=50.0)
    )

    # rebalance policy
    train_rebalance: RebalanceConfig = field(
        default_factory=lambda: RebalanceConfig(_strategy="directed", max_trials=100)
    )
    test_rebalance: RebalanceConfig = field(
        default_factory=lambda: RebalanceConfig(
            _strategy="deterministic", max_trials=100
        )
    )

    # Reset, observation, agent
    _reset: ResetConfig = field(default_factory=ResetConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Episode
    num_steps_per_episode: int = 1000

    def from_dict(self, config: dict[str, Any]) -> None:
        self.swing.from_dict(config.pop("swing"))

        self.reward.from_dict(config.pop("reward"))
        self.reward_failed.from_dict(config.pop("reward_failed"))
        self.reward_failed_rebalance.from_dict(config.pop("reward_failed_rebalance"))
        self.reward_failed_steady.from_dict(config.pop("reward_failed_steady"))

        self.train_rebalance.from_dict(config.pop("train_rebalance"))
        self.test_rebalance.from_dict(config.pop("test_rebalance"))

        self.reset = ResetConfig(**config.pop("reset"))
        self.observation.from_dict(config.pop("observation"))
        self.agent.from_dict(config.pop("agent"))

        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    # --------------------------- Reset ------------------------------
    @property
    def reset(self) -> ResetConfig:
        return self._reset

    @reset.setter
    def reset(self, reset: dict[str, bool] | ResetConfig) -> None:
        if isinstance(reset, dict):
            assert list(reset.keys()) == ["graph", "coupling", "node_type", "node"]
            reset = ResetConfig(**reset)
        self._reset = reset


RL_CONFIG = RLConfig()
