"""
DummyVecEnv implementation of SB3
- All unneccessary codes of SMUD project are removed
- #! comments are from HY
"""

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Iterable, Sequence, Type

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Define type aliases here to avoid circular import

# Used when we want to access one or more VecEnv
VecEnvIndices = int | Iterable[int] | None

# VecEnvObs is what is returned by the reset() method
# it contains the observation for each env
VecEnvObs = np.ndarray | dict[str, np.ndarray] | tuple[np.ndarray, ...]

# VecEnvStepReturn is what is returned by the step() method
# it contains the observation, reward, done, info for each env
VecEnvStepReturn = tuple[VecEnvObs, np.ndarray, np.ndarray, list[dict]]


def obs_space_info(
    obs_space: spaces.Space,
) -> tuple[list[str], dict[Any, tuple[int, ...]], dict[Any, np.dtype]]:
    """
    Get dict-structured information about a gym.Space.

    Dict spaces are represented directly by their dict of subspaces.
    Tuple spaces are converted into a dict with keys indexing into the tuple.
    Unstructured spaces are represented by {None: obs_space}.

    :param obs_space: an observation space
    :return: A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """

    # Make sure the observation space does not have nested spaces (Dicts/Tuples inside Dicts/Tuples).
    # If so, raise an Exception informing that there is no support for this.
    if isinstance(obs_space, (spaces.Dict, spaces.Tuple)):
        sub_spaces = (
            obs_space.spaces.values()
            if isinstance(obs_space, spaces.Dict)
            else obs_space.spaces
        )
        for sub_space in sub_spaces:
            if isinstance(sub_space, (spaces.Dict, spaces.Tuple)):
                raise NotImplementedError(
                    "Nested observation spaces are not supported (Tuple/Dict space"
                    " inside Tuple/Dict space)."
                )

    if isinstance(obs_space, spaces.Dict):
        assert isinstance(
            obs_space.spaces, OrderedDict
        ), "Dict space must have ordered subspaces"
        subspaces = obs_space.spaces
    elif isinstance(obs_space, spaces.Tuple):
        subspaces = {i: space for i, space in enumerate(obs_space.spaces)}  # type: ignore[assignment]
    else:
        assert not hasattr(
            obs_space, "spaces"
        ), f"Unsupported structured space '{type(obs_space)}'"
        subspaces = {None: obs_space}  # type: ignore[assignment]
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape  # type: ignore
        dtypes[key] = box.dtype  # type: ignore
    return keys, shapes, dtypes


def is_wrapped(env: gym.Env, wrapper_class: Type[gym.Wrapper]) -> bool:
    """
    Check if a given environment has been wrapped with a given wrapper.

    :param env: Environment to check
    :param wrapper_class: Wrapper class to look for
    :return: True if environment has been wrapped with ``wrapper_class``.
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return True
        env_tmp = env_tmp.env
    return False


class DummyVecEnv:
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]]):
        #! Create multiple environments
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create"
                " them returned the same instance instead of creating different"
                " objects. You are probably using `make_vec_env(lambda: env)` or"
                " `DummyVecEnv([lambda: env] * n_envs)`. You should replace `lambda:"
                " env` by a `make_env` function that creates a new instance of the"
                " environment at every call (using `gym.make()` for instance). You can"
                " take a look at the documentation for an example. Please read"
                " https://github.com/DLR-RM/stable-baselines3/issues/1151 for more"
                " information."
            )

        #! Extract information from single environment
        env = self.envs[0]
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = env.metadata

        #! Inspect observation space
        self.keys, shapes, dtypes = obs_space_info(self.observation_space)

        #! Buffers: for rollout (multiple steps)
        self.buf_obs = OrderedDict(
            [
                (k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k]))
                for k in self.keys
            ]
        )
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        #! reset of each environment
        # store info returned by the reset method
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        # seeds to be used in the next call to env.reset()
        self._seeds: list[int | None] = [None for _ in range(self.num_envs)]
        # options to be used in the next call to env.reset()
        self._options: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        #! Ignore render_mode
        self.render_mode = None

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        for env_idx, env in enumerate(self.envs):
            #! Step for every environments
            obs, rews, terminated, truncated, infos = env.step(actions[env_idx])

            #! Get results of step
            self.buf_rews[env_idx] = rews
            self.buf_infos[env_idx] = infos
            # convert to SB3 VecEnv api
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_dones[env_idx] = terminated or truncated
            self.buf_infos[env_idx]["TimeLimit.truncated"] = (
                truncated and not terminated
            )

            #! If done
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = env.reset()

            #! Store observation to self.buf_obs
            self._save_obs(env_idx, obs)

        return (
            self._obs_from_buf(),  #! Read current observation from self.buf_obs
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            maybe_options = (
                {"options": self._options[env_idx]} if self._options[env_idx] else {}
            )
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(
                seed=self._seeds[env_idx], **maybe_options
            )

            #! Store observation of reset to self.buf_obs
            self._save_obs(env_idx, obs)

        # Seeds and options are only used once
        self._seeds = [None for _ in range(self.num_envs)]
        self._options = [{} for _ in range(self.num_envs)]

        #! Read reset observation. Info is not returned: Different from gymnasium.Env api
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]  # type: ignore[call-overload]

    def _obs_from_buf(self) -> VecEnvObs:
        #! Read current observation from self.buf_obs
        return OrderedDict([(k, np.copy(v)) for k, v in self.buf_obs.items()])

    def _get_target_envs(self, indices: VecEnvIndices) -> list[gym.Env]:
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.
        Return the list of envs corresponding to the indices.

        :param indices: refers to indices of envs.
        :return: the implied lists of environments.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]

        return [self.envs[i] for i in indices]

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> list[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> list[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def seed(self, seed: int | None = None) -> Sequence[int | None]:
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.
        WARNING: since gym 0.26, those seeds will only be passed to the environment
        at the next reset.

        :param seed: The random seed. May be None for completely random seeding.
        :return: Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        if seed is None:
            # To ensure that subprocesses have different seeds,
            # we still populate the seed variable when no argument is passed

            #! You need to call np.random.seed before running SB3 for reproducibility
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))

        self._seeds = [seed + idx for idx in range(self.num_envs)]
        return self._seeds

    def set_options(self, options: dict | list[dict] | None = None) -> None:
        """
        Set environment options for all environments.
        If a dict is passed instead of a list, the same options will be used for all environments.
        WARNING: Those options will only be passed to the environment at the next reset.

        :param options: A dictionary of environment options to pass to each environment at the next reset.
        """
        if options is None:
            options = {}
        # Use deepcopy to avoid side effects
        if isinstance(options, dict):
            self._options = deepcopy([options] * self.num_envs)
        else:
            self._options = deepcopy(options)
