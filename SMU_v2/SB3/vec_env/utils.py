from collections import OrderedDict
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from SB3.type_aliases import VecEnvObs

from .base_vec_env import VecEnv, VecEnvWrapper


def check_for_nested_spaces(obs_space: spaces.Dict) -> None:
    """
    Make sure the observation space does not have nested spaces (Dicts/Tuples inside Dicts/Tuples).
    If so, raise an Exception informing that there is no support for this.

    :param obs_space: an observation space
    """
    for sub_space in obs_space.spaces.values():
        if isinstance(sub_space, (spaces.Dict, spaces.Tuple)):
            raise NotImplementedError(
                "Nested observation spaces are not supported (Tuple/Dict space inside"
                " Tuple/Dict space)."
            )


def obs_space_info(
    obs_space: spaces.Dict,
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

    #! Sanity check: no nested spaces
    check_for_nested_spaces(obs_space)

    keys: list[str] = []
    shapes = {}
    dtypes = {}
    for key, box in obs_space.spaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    raise NotImplementedError(f"{observation_space} observation space is not supported")


def preprocess_obs(
    obs: torch.Tensor | dict[str, torch.Tensor],
    observation_space: spaces.Space,
    normalize_images: bool = True,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """
    Preprocess observation to be to a neural network.
    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
    For discrete observations, it create a one hot vector.

    :param obs: Observation
    :param observation_space:
    :param normalize_images: Whether to normalize images or not
        (True by default)
    :return:
    """
    if isinstance(observation_space, spaces.Dict):
        # Do not modify by reference the original observation
        assert isinstance(obs, dict), f"Expected dict, got {type(obs)}"
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs(_obs, observation_space[key], normalize_images=normalize_images)
        return preprocessed_obs

    assert isinstance(obs, torch.Tensor), f"Expecting a torch Tensor, but got {type(obs)}"

    if isinstance(observation_space, spaces.Box):
        return obs.float()

    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return F.one_hot(obs.long(), num_classes=int(observation_space.n)).float()

    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        return torch.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
                for idx, obs_ in enumerate(torch.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.float()
    else:
        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")

def get_action_dim(action_space: spaces.Box) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    return int(np.prod(action_space.shape))

def unwrap_wrapper(
    env: gym.Env, wrapper_class: type[gym.Wrapper]
) -> gym.Wrapper | None:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env: Environment to unwrap
    :param wrapper_class: Wrapper to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env: gym.Env, wrapper_class: type[gym.Wrapper]) -> bool:
    """
    Check if a given environment has been wrapped with a given wrapper.

    :param env: Environment to check
    :param wrapper_class: Wrapper class to look for
    :return: True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None


def unwrap_vec_wrapper(
    env: VecEnv, vec_wrapper_class: type[VecEnvWrapper]
) -> VecEnvWrapper | None:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env: The ``VecEnv`` that is going to be unwrapped
    :param vec_wrapper_class: The desired ``VecEnvWrapper`` class.
    :return: The ``VecEnvWrapper`` object if the ``VecEnv`` is wrapped with the desired wrapper, None otherwise
    """
    env_tmp = env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, vec_wrapper_class):
            return env_tmp
        env_tmp = env_tmp.venv
    return None



def flatten_obs(
    obs: list[VecEnvObs] | tuple[VecEnvObs], space: spaces.Dict
) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    return OrderedDict(
        [(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()]
    )


def check_shape_equal(space1: spaces.Space, space2: spaces.Space) -> None:
    """
    If the spaces are Box, check that they have the same shape.

    If the spaces are Dict, it recursively checks the subspaces.

    :param space1: Space
    :param space2: Other space
    """
    if isinstance(space1, spaces.Dict):
        assert isinstance(space2, spaces.Dict), "spaces must be of the same type"
        assert (
            space1.spaces.keys() == space2.spaces.keys()
        ), "spaces must have the same keys"
        for key in space1.spaces.keys():
            check_shape_equal(space1.spaces[key], space2.spaces[key])

    elif isinstance(space1, spaces.Box):
        assert space1.shape == space2.shape, "spaces must have the same shape"


def check_for_correct_spaces(
    env: gym.Env | VecEnv, observation_space: spaces.Space, action_space: spaces.Space
) -> None:
    """
    Checks that the environment has same spaces as provided ones. Used by BaseAlgorithm to check if
    spaces match after loading the model with given env.
    Checked parameters:
    - observation_space
    - action_space

    :param env: Environment to check for valid spaces
    :param observation_space: Observation space to check against
    :param action_space: Action space to check against
    """
    if observation_space != env.observation_space:
        raise ValueError(
            f"Observation spaces do not match: {observation_space} !="
            f" {env.observation_space}"
        )
    if action_space != env.action_space:
        raise ValueError(
            f"Action spaces do not match: {action_space} != {env.action_space}"
        )



def is_vectorized_box_observation(observation: np.ndarray, observation_space: spaces.Box) -> bool:
    """
    For box observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    if observation.shape == observation_space.shape:
        return False
    elif observation.shape[1:] == observation_space.shape:
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for "
            + f"Box environment, please use {observation_space.shape} "
            + "or (n_env, {}) for the observation shape.".format(", ".join(map(str, observation_space.shape)))
        )


def is_vectorized_discrete_observation(observation: int | np.ndarray, observation_space: spaces.Discrete) -> bool:
    """
    For discrete observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    if isinstance(observation, int) or observation.shape == ():  # A numpy array of a number, has shape empty tuple '()'
        return False
    elif len(observation.shape) == 1:
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for "
            + "Discrete environment, please use () or (n_env,) for the observation shape."
        )


def is_vectorized_multidiscrete_observation(observation: np.ndarray, observation_space: spaces.MultiDiscrete) -> bool:
    """
    For multidiscrete observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    if observation.shape == (len(observation_space.nvec),):
        return False
    elif len(observation.shape) == 2 and observation.shape[1] == len(observation_space.nvec):
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for MultiDiscrete "
            + f"environment, please use ({len(observation_space.nvec)},) or "
            + f"(n_env, {len(observation_space.nvec)}) for the observation shape."
        )


def is_vectorized_multibinary_observation(observation: np.ndarray, observation_space: spaces.MultiBinary) -> bool:
    """
    For multibinary observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    if observation.shape == observation_space.shape:
        return False
    elif len(observation.shape) == len(observation_space.shape) + 1 and observation.shape[1:] == observation_space.shape:
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for MultiBinary "
            + f"environment, please use {observation_space.shape} or "
            + f"(n_env, {observation_space.n}) for the observation shape."
        )


def is_vectorized_dict_observation(observation: np.ndarray, observation_space: spaces.Dict) -> bool:
    """
    For dict observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    # We first assume that all observations are not vectorized
    all_non_vectorized = True
    for key, subspace in observation_space.spaces.items():
        # This fails when the observation is not vectorized
        # or when it has the wrong shape
        if observation[key].shape != subspace.shape:
            all_non_vectorized = False
            break

    if all_non_vectorized:
        return False

    all_vectorized = True
    # Now we check that all observation are vectorized and have the correct shape
    key = ""    #! Unbound type-error
    for key, subspace in observation_space.spaces.items():
        if observation[key].shape[1:] != subspace.shape:
            all_vectorized = False
            break

    if all_vectorized:
        return True
    else:
        # Retrieve error message
        error_msg = ""
        try:
            is_vectorized_observation(observation[key], observation_space.spaces[key])
        except ValueError as e:
            error_msg = f"{e}"
        raise ValueError(
            f"There seems to be a mix of vectorized and non-vectorized observations. "
            f"Unexpected observation shape {observation[key].shape} for key {key} "
            f"of type {observation_space.spaces[key]}. {error_msg}"
        )




def is_vectorized_observation(observation: int | np.ndarray, observation_space: spaces.Space) -> bool:
    """
    For every observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """

    is_vec_obs_func_dict = {
        spaces.Box: is_vectorized_box_observation,
        spaces.Discrete: is_vectorized_discrete_observation,
        spaces.MultiDiscrete: is_vectorized_multidiscrete_observation,
        spaces.MultiBinary: is_vectorized_multibinary_observation,
        spaces.Dict: is_vectorized_dict_observation,
    }

    for space_type, is_vec_obs_func in is_vec_obs_func_dict.items():
        if isinstance(observation_space, space_type):
            return is_vec_obs_func(observation, observation_space)

    # for-else happens if no break is called
    raise ValueError(f"Error: Cannot determine if the observation is vectorized with the space type {observation_space}.")