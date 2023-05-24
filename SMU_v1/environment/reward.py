from typing import Protocol

import numpy as np
import numpy.typing as npt
from config import RL_CONFIG, SWING_CONFIG

DTYPE = SWING_CONFIG.dtype


class Reward(Protocol):
    def __call__(self, dphases: npt.NDArray[DTYPE], times: npt.NDArray[DTYPE]) -> float:
        ...


class SlopeReward:
    """
    Reward = slope of the initial fluctuation \\
    R ~ [1/N sum_i abs(dphase_i[1] - dphase_i[0])] / dt

    Args
        dphases: [S, N]
        time: Not used
    Return
        reward: negative value, higher the better
    """
    def __call__(self, dphases: npt.NDArray[DTYPE], times: npt.NDArray[DTYPE]) -> float:
        return -(np.mean(np.abs(dphases[1] - dphases[0])) / SWING_CONFIG.dt).item()


class AreaReward:
    """
    Reward = area under dphases graph \\
    R ~ 1/T sum_t [1/N sum_i abs(dphase_i)]

    Args
        dphases: [S, N]
        times: Not used
    Return
        reward: negative value, higher the better
    """
    def __call__(self, dphases: npt.NDArray[DTYPE], times: npt.NDArray[DTYPE]) -> float:
        return -np.mean(np.abs(dphases)).item()


class WeightedAreaReward:
    """
    Reward = area under dphases graph, weighted by it's time \\
    R ~ 1/T sum_t t * [1/N sum_i abs(dphase_i)]

    Args
        dphases: [S, N]
        time: [S, ]
    Return
        reward: negative value, higher the better
    """
    def __call__(self, dphases: npt.NDArray[DTYPE], times: npt.NDArray[DTYPE]) -> float:
        return -np.mean(np.abs(dphases).mean(axis=1) * times).item()


class ThresholdAreaReward:
    """
    Reward = area of dphases graph, in which exceeds certain threshold \\
    R ~ 1/T sum_t [1/N sum_i max(0.0, abs(dphase_i) - threshold)]

    Args
        dphases: [S, N]
        time: Not used
    Return
        reward: negative value, higher the better
    """
    def __call__(self, dphases: npt.NDArray[DTYPE], times: npt.NDArray[DTYPE]) -> float:
        area = np.maximum(
            np.zeros_like(dphases), np.abs(dphases) - RL_CONFIG.fail_threshold
        )
        return -np.mean(area).item()


class WeightedThresholdAreaReward:
    """
    Reward = area of dphases graph, in which exceeds certain threshold weighted by time \\
    R ~ 1/T sum_t [1/N sum_i t * max(0.0, abs(dphase_i) - threshold)]

    Args
        dphases: [S, N]
        time: [S, ]
    Return
        reward: negative value, higher the better
    """
    def __call__(self, dphases: npt.NDArray[DTYPE], times: npt.NDArray[DTYPE]) -> float:
        area = np.maximum(
            np.zeros_like(dphases), np.abs(dphases) - RL_CONFIG.fail_threshold
        )
        return -np.mean(area.mean(axis=1) * times).item()


def get_reward_ftn() -> Reward:
    if RL_CONFIG.reward == "area":
        return AreaReward()
    elif RL_CONFIG.reward == "slope":
        return SlopeReward()
    elif RL_CONFIG.reward == "weighted_area":
        return WeightedAreaReward()
    elif RL_CONFIG.reward == "threshold_area":
        return ThresholdAreaReward()
    elif RL_CONFIG.reward == "weighted_threshold_area":
        return WeightedThresholdAreaReward()
    raise ValueError(f"No such reward: {RL_CONFIG.reward}")


def reward_failed(num_failed: int, time: float) -> float:
    """
    Reward when encountering failed nodes
    num_failed: number of failed nodes
    time: time when node is failed

    R ~ num/time
    """
    return -RL_CONFIG.failed_scale * num_failed / time
