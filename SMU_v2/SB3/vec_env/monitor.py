"""
Monitor wrapper implementation of SB3
- All unneccessary codes of SMUD project are removed
- #! comments are from HY
"""

import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class Monitor(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """

    EXT = "monitor.csv"

    def __init__(self, env: gym.Env):
        super().__init__(env=env)
        self.t_start = time.time()

        self.rewards: list[float] = []
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_times: list[float] = []

        self.needs_reset = True
        self.total_steps = 0

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        """
        Calls the Gym environment reset. Can only be called if the environment is over.

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        self.rewards = []
        self.needs_reset = False
        return self.env.reset(**kwargs)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))

        if terminated or truncated:
            self.needs_reset = True

            #! Get reward, length, time of current episode
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {
                "r": round(ep_rew, 6),
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6),
            }

            #! Store data
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            info["episode"] = ep_info

        self.total_steps += 1
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """
        Closes the environment
        """
        super().close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> list[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> list[int]:
        """
        Returns the number of timesteps of all the episodes

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> list[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return:
        """
        return self.episode_times
