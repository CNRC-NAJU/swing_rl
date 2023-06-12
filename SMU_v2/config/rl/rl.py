from dataclasses import dataclass
from typing import Any

from config.grid import RebalanceConfig, ResetConfig, SwingConfig

from .agent import AgentConfig
from .observation import ObservationConfig
from .reward import RewardConfig


@dataclass(slots=True)
class RLConfig:
    # Trajectory for checking failed nodes, calculating reward
    swing: SwingConfig = SwingConfig(
        _solver_name="rk4",
        dt=1e-2,
        max_time=2.0,
        _monitor_name="outside",
        monitor_threshold=1.0,
    )

    # Reward functions
    reward: RewardConfig = RewardConfig(_name="weighted_area", threshold=0.0)
    reward_failed: RewardConfig = RewardConfig(_name="inverse_time", scale=10.0)
    reward_failed_rebalance: RewardConfig = RewardConfig(_name="constant", scale=100.0)
    reward_failed_steady: RewardConfig = RewardConfig(_name="constant", scale=50.0)

    # rebalance policy
    train_rebalance: RebalanceConfig = RebalanceConfig(
        _strategy="directed", max_trials=100
    )
    test_rebalance: RebalanceConfig = RebalanceConfig(
        _strategy="deterministic", max_trials=100
    )

    # Reset, observation, agent
    reset: ResetConfig = ResetConfig()
    observation: ObservationConfig = ObservationConfig()
    agent: AgentConfig = AgentConfig()

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


RL_CONFIG = RLConfig()
