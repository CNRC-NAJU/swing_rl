from typing import Protocol

import numpy as np
import numpy.typing as npt
from config.rl import RewardConfig


class RewardFtn(Protocol):
    def __call__(self, dphases: npt.NDArray[np.float64], dt: float) -> float:
        ...


class SlopeRewardFtn:
    """
    Reward = slope of the dphase at initial fluctuation \\
    R ~ [1/N sum_i abs(dphase_i[1] - dphase_i[0])] / dt

    Args
        dphases: [S+1, N]
        time: Not used
    Return
        reward: negative value, higher the better
    """

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def __call__(self, dphases: npt.NDArray[np.float64], dt: float) -> float:
        # Initial acceleration: [N, ]
        initial_acceleration = np.abs(dphases[1] - dphases[0])

        return -self.scale * np.mean(initial_acceleration).item() / dt


class AreaRewardFtn:
    """
    Reward = area under dphases graph over time \\
    R ~ 1/T sum_t [1/N sum_i max(0.0, abs(dphase_i) - threshold)]

    if threshold=0, simply R ~ 1/T sum_t [1/N sum_i abs(dphase_i)]

    Args
        dphases: [S+1, N]
        dt: Not used
    Return
        reward: negative value, higher the better
    """

    def __init__(self, scale: float, threshold: float) -> None:
        self.scale = scale
        self.threshold = threshold

    def __call__(self, dphases: npt.NDArray[np.float64], dt: float) -> float:
        # Average height of dphases; [S+1, N] -> [S+1, ]
        height = np.clip(np.abs(dphases) - self.threshold, 0.0, None).mean(axis=1)

        # Average over time
        return -self.scale * np.mean(height).item() * dt


class WeightedAreaRewardFtn:
    """
    Reward = area under dphases graph over time, weighted by time \\
    R ~ 1/T sum_t [1/N sum_i t * max(0.0, abs(dphase_i) - threshold)]

    if threshold=0, simply R ~ 1/T sum_t t * [1/N sum_i abs(dphase_i)]

    Args
        dphases: [S+1, N]
        dt: time delta
    Return
        reward: negative value, higher the better
    """

    def __init__(self, scale: float, threshold: float) -> None:
        self.scale = scale
        self.threshold = threshold

    def __call__(self, dphases: npt.NDArray[np.float64], dt: float) -> float:
        # Average height of dphases; [S+1, N] -> [S+1, ]
        height = np.clip(np.abs(dphases) - self.threshold, 0.0, None).mean(axis=1)
        time = np.arange(len(dphases)) * dt

        return -self.scale * np.mean(height * time).item() * dt


class InverseTimeRewardFtn:
    """
    Reward = proportional to 1/T where T is time

    Note: given dt is not time interval. It is T (time when reward is called)
    """

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def __call__(self, dphases: npt.NDArray[np.float64], dt: float) -> float:
        reward_time = dt
        return -1 / reward_time


class ConstantRewardFtn:
    """Return constant value"""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def __call__(self, dphases: npt.NDArray[np.float64], dt: float) -> float:
        return -self.scale


def get_reward_ftn(config: RewardConfig) -> RewardFtn:
    match config.name:
        case "area":
            return AreaRewardFtn(config.scale, config.threshold)
        case "weighted_area":
            return WeightedAreaRewardFtn(config.scale, config.threshold)
        case "slope":
            return SlopeRewardFtn(config.scale)
        case "inverse_time":
            return InverseTimeRewardFtn(config.scale)
        case "constant":
            return ConstantRewardFtn(config.scale)
        case _:
            raise TypeError(f"No such reward: {config.name}")
