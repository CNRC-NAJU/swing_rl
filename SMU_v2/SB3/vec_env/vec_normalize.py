"""
VecNormalize implementation of SB3
- All unneccessary codes of SMUD project are removed
- #! comments are from HY
"""

from __future__ import annotations

import inspect
import pickle
from copy import deepcopy
from typing import Any

import numpy as np
from gymnasium import spaces
from SB3.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from SB3.type_aliases import VecEnvObs, VecEnvStepReturn
from SB3.vec_env.utils import check_shape_equal


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def copy(self) -> RunningMeanStd:
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: RunningMeanStd) -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class VecNormalize(VecEnvWrapper):
    """
    A moving average, normalizing wrapper for vectorized environment.
    has support for saving/loading moving average,

    :param venv: the vectorized environment to wrap
    :param training: Whether to update or not the moving average
    :param norm_obs: Whether to normalize observation or not (default: True)
    :param norm_reward: Whether to normalize rewards or not (default: True)
    :param clip_obs: Max absolute value for observation
    :param clip_reward: Max value absolute for discounted reward
    :param gamma: discount factor
    :param epsilon: To avoid division by zero
    :param norm_obs_keys: Which keys from observation dict to normalize.
        If not specified, all keys will be normalized.
    """

    venv: VecEnv
    observation_space: spaces.Dict
    action_space: spaces.Box
    obs_spaces: dict[str, spaces.Space]
    old_obs: VecEnvObs

    def __init__(
        self,
        venv: VecEnv,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        norm_obs_keys: list[str] | None = None,
    ):
        VecEnvWrapper.__init__(self, venv)

        self.norm_obs = norm_obs
        self.norm_obs_keys = norm_obs_keys
        # Check observation spaces
        if self.norm_obs:
            # By default, we normalize all keys
            if self.norm_obs_keys is None:
                self.norm_obs_keys = list(self.observation_space.spaces.keys())

            # Check that all keys are of type Box
            for obs_key in self.norm_obs_keys:
                if not isinstance(self.observation_space.spaces[obs_key], spaces.Box):
                    raise ValueError(
                        "VecNormalize only supports `gym.spaces.Box` observation"
                        f" spaces but {obs_key} is of type"
                        f" {self.observation_space.spaces[obs_key]}. You should"
                        " probably explicitely pass the observation keys  that should"
                        " be normalized via the `norm_obs_keys` parameter."
                    )

            self.obs_spaces = self.observation_space.spaces
            self.obs_rms = {key: RunningMeanStd(shape=self.obs_spaces[key].shape) for key in self.norm_obs_keys}  # type: ignore[arg-type, union-attr]

        self.ret_rms = RunningMeanStd()
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.returns = np.zeros(self.num_envs)  # Returns: discounted rewards
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.old_reward = np.array([])

    def __getstate__(self) -> dict[str, Any]:
        """
        Gets state for pickling.

        Excludes self.venv, as in general VecEnv's may not be pickleable."""
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state["venv"]
        del state["class_attributes"]
        # these attributes depend on the above and so we would prefer not to pickle
        del state["returns"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restores pickled state.

        User must call set_venv() after unpickling before using.

        :param state:"""
        # Backward compatibility
        if "norm_obs_keys" not in state and isinstance(
            state["observation_space"], spaces.Dict
        ):
            state["norm_obs_keys"] = list(state["observation_space"].spaces.keys())
        self.__dict__.update(state)
        assert "venv" not in state
        self.venv = None  # type: ignore[assignment]

    def set_venv(self, venv: VecEnv) -> None:
        """
        Sets the vector environment to wrap to venv.

        Also sets attributes derived from this such as `num_env`.

        :param venv:
        """
        if self.venv is not None:
            raise ValueError(
                "Trying to set venv of already initialized VecNormalize wrapper."
            )
        self.venv = venv
        self.num_envs = venv.num_envs
        self.class_attributes = dict(inspect.getmembers(self.__class__))

        # Check that the observation_space shape match
        check_shape_equal(self.observation_space, venv.observation_space)
        self.returns = np.zeros(self.num_envs)

    def step_wait(self) -> VecEnvStepReturn:
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, dones)

        where ``dones`` is a boolean vector indicating whether each element is new.
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        self.old_obs = obs
        self.old_reward = rewards

        if self.training and self.norm_obs:
            for key in self.obs_rms.keys():
                self.obs_rms[key].update(obs[key])

        obs = self.normalize_obs(obs)

        if self.training:
            self._update_reward(rewards)
        rewards = self.normalize_reward(rewards)

        # Normalize the terminal observations
        for idx, done in enumerate(dones):
            if not done:
                continue
            if "terminal_observation" in infos[idx]:
                infos[idx]["terminal_observation"] = self.normalize_obs(
                    infos[idx]["terminal_observation"]
                )

        self.returns[dones] = 0
        return obs, rewards, dones, infos

    def _update_reward(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        self.returns = self.returns * self.gamma + reward
        self.ret_rms.update(self.returns)

    def normalize_obs(self, obs: VecEnvObs) -> VecEnvObs:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """

        def _normalize(obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
            return np.clip(
                (obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            )

        # Avoid modifying by reference the original object
        obs_ = deepcopy(obs)
        if self.norm_obs:
            assert self.norm_obs_keys is not None
            # Only normalize the specified keys
            for key in self.norm_obs_keys:
                obs_[key] = _normalize(obs[key], self.obs_rms[key]).astype(np.float32)
        return obs_

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """
        Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.
        """
        if self.norm_reward:
            reward = np.clip(
                reward / np.sqrt(self.ret_rms.var + self.epsilon),
                -self.clip_reward,
                self.clip_reward,
            )
        return reward

    def unnormalize_obs(self, obs: VecEnvObs) -> VecEnvObs:
        def _unnormalize(obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
            """
            Helper to unnormalize observation.
            :param obs:
            :param obs_rms: associated statistics
            :return: unnormalized observation
            """
            return (obs * np.sqrt(obs_rms.var + self.epsilon)) + obs_rms.mean

        # Avoid modifying by reference the original object
        obs_ = deepcopy(obs)
        if self.norm_obs:
            assert self.norm_obs_keys is not None
            for key in self.norm_obs_keys:
                obs_[key] = _unnormalize(obs[key], self.obs_rms[key])
        return obs_

    def unnormalize_reward(self, reward: np.ndarray) -> np.ndarray:
        if self.norm_reward:
            return reward * np.sqrt(self.ret_rms.var + self.epsilon)
        return reward

    def get_original_obs(self) -> VecEnvObs:
        """
        Returns an unnormalized version of the observations from the most recent
        step or reset.
        """
        return deepcopy(self.old_obs)

    def get_original_reward(self) -> np.ndarray:
        """
        Returns an unnormalized version of the rewards from the most recent step.
        """
        return self.old_reward.copy()

    def reset(self) -> VecEnvObs:
        """
        Reset all environments
        :return: first observation of the episode
        """
        obs = self.venv.reset()
        self.old_obs = obs
        self.returns = np.zeros(self.num_envs)

        if self.training and self.norm_obs:
            for key in self.obs_rms.keys():
                self.obs_rms[key].update(obs[key])

        return self.normalize_obs(obs)

    @staticmethod
    def load(load_path: str, venv: VecEnv) -> VecNormalize:
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return:
        """
        with open(load_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        vec_normalize.set_venv(venv)
        return vec_normalize

    def save(self, save_path: str) -> None:
        """
        Save current VecNormalize object with
        all running statistics and settings (e.g. clip_obs)

        :param save_path: The path to save to
        """
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)
