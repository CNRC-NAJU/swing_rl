from typing import Protocol

import numpy as np
import numpy.typing as npt
from config import RLConfig, SwingConfig


class Reward(Protocol):
    def __call__(
        self,
        dphases: npt.NDArray[SwingConfig.dtype],
        times: npt.NDArray[SwingConfig.dtype],
    ) -> float:
        ...


class AreaReward:
    def __call__(
        self,
        dphases: npt.NDArray[SwingConfig.dtype],
        times: npt.NDArray[SwingConfig.dtype],
    ) -> float:
        """
        R ~ 1/T sum_t [1/N sum_i abs(dphase_i)] * dt
        dphases: [S, N]
        """
        return -np.mean(np.abs(dphases)).item() * SwingConfig._dt


class SlopeReward:
    def __call__(
        self,
        dphases: npt.NDArray[SwingConfig.dtype],
        times: npt.NDArray[SwingConfig.dtype],
    ) -> float:
        """
        R ~ [1/N sum_i abs(dphase_i[1] - dphase_i[0])] / dt
        dphases: [S, N]
        """
        return -np.mean(np.abs(dphases[1] - dphases[0])).item() / SwingConfig._dt


class WeightedAreaReward:
    def __call__(
        self,
        dphases: npt.NDArray[SwingConfig.dtype],
        times: npt.NDArray[SwingConfig.dtype],
    ) -> float:
        """
        R ~ 1/T sum_t [1/N sum_i t * abs(dphase_i)] * dt
        dphases: [S, N]
        time: [S, ]
        """
        return -np.mean(np.abs(dphases).mean(axis=1) * times).item()


def get_reward_ftn() -> Reward:
    if RLConfig.reward == "area":
        return AreaReward()
    elif RLConfig.reward == "slope":
        return SlopeReward()
    elif RLConfig.reward == "weighted_area":
        return WeightedAreaReward()
    else:
        raise ValueError(f"No such reward: {RLConfig.reward}")


def reward_failed(num_failed: int, time: float) -> float:
    """
    Reward when encountering failed nodes
    num_failed: number of failed nodes
    time: time when node is failed

    R ~ num/time
    """
    return -RLConfig.failed_scale * num_failed / time
