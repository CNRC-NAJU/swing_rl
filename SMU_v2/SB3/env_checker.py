import warnings
from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .vec_env import DummyVecEnv, VecCheckNan, check_for_nested_spaces


def check_env(env: gym.Env, warn: bool = True, skip_render_check: bool = True) -> None:
    """
    Check that an environment follows Gym API.
    This is particularly useful when using a custom environment.
    Please take a look at https://gymnasium.farama.org/api/env/
    for more information about the API.

    It also optionally check that the environment is compatible with Stable-Baselines.

    :param env: The Gym environment that will be checked
    :param warn: Whether to output additional warnings
        mainly related to the interaction with Stable Baselines
    :param skip_render_check: Whether to skip the checks for the render method.
        True by default (useful for the CI)
    """
    assert isinstance(env, gym.Env), (
        "Your environment must inherit from the gymnasium.Env class cf."
        " https://gymnasium.farama.org/api/env/"
    )

    # ============= Check the spaces (observation and action) ================
    _check_spaces(env)

    # Define aliases for convenience
    observation_space = cast(spaces.Dict, env.observation_space)
    action_space = cast(spaces.Box, env.action_space)

    try:
        env.reset(seed=0)
    except TypeError as e:
        raise TypeError("The reset() method must accept a `seed` parameter") from e

    # Warn the user if needed.
    # A warning means that the environment may run but not work properly with Stable Baselines algorithms
    if warn:
        check_for_nested_spaces(observation_space)
        _check_unsupported_spaces(env, observation_space, action_space)

        for key, space in observation_space.spaces.items():
            if isinstance(space, spaces.Box):
                _check_box_obs(space, key)

        # Check for the action space, it may lead to hard-to-debug issues
        assert np.all(
            np.isfinite(np.array([action_space.low, action_space.high]))
        ), "Continuous action space must have a finite lower and upper bound"

        if (
            np.any(np.abs(action_space.low) != np.abs(action_space.high))
            or np.any(action_space.low != -1)
            or np.any(action_space.high != 1)
        ):
            warnings.warn(
                "We recommend you to use a symmetric and normalized Box action space"
                " (range=[-1, 1]) cf."
                " https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html"
            )

        if action_space.dtype != np.dtype(np.float32):
            warnings.warn(
                f"Your action space has dtype {action_space.dtype}, we recommend using"
                " np.float32 to avoid cast errors."
            )

    # ============ Check the returned values ===============
    _check_returned_values(env, observation_space, action_space)

    _check_nan(env)


def _check_unsupported_spaces(
    env: gym.Env, observation_space: spaces.Dict, action_space: spaces.Box
) -> None:
    """Emit warnings when the observation space or action space used is not supported by Stable-Baselines."""

    nested_dict = False
    for key, space in observation_space.spaces.items():
        if isinstance(space, spaces.Dict):
            nested_dict = True
        if isinstance(space, spaces.Discrete) and space.start != 0:
            warnings.warn(
                f"Discrete observation space (key '{key}') with a non-zero start is not"
                " supported by Stable-Baselines3. You can use a wrapper or update your"
                " observation space."
            )

    if nested_dict:
        warnings.warn(
            "Nested observation spaces are not supported by Stable Baselines3 (Dict"
            " spaces inside Dict space). You should flatten it to have only one level"
            " of keys.For example, `dict(space1=dict(space2=Box(), space3=Box()),"
            " spaces4=Discrete())` is not supported but `dict(space2=Box(),"
            " spaces3=Box(), spaces4=Discrete())` is."
        )


def _check_nan(env: gym.Env) -> None:
    """Check for Inf and NaN using the VecWrapper."""
    vec_env = VecCheckNan(DummyVecEnv([lambda: env]))
    vec_env.reset()
    for _ in range(10):
        action = np.array([env.action_space.sample()])
        _, _, _, _ = vec_env.step(action)


def _check_obs(
    obs: tuple | dict | np.ndarray | int,
    observation_space: spaces.Space,
    method_name: str,
) -> None:
    """
    Check that the observation returned by the environment
    correspond to the declared one.
    """
    if not isinstance(observation_space, spaces.Tuple):
        assert not isinstance(obs, tuple), (
            f"The observation returned by the `{method_name}()` method should be a"
            " single value, not a tuple"
        )

    # The check for a GoalEnv is done by the base class
    if isinstance(observation_space, spaces.Discrete):
        # Since https://github.com/Farama-Foundation/Gymnasium/pull/141,
        # `sample()` will return a np.int64 instead of an int
        assert np.issubdtype(
            type(obs), np.integer
        ), f"The observation returned by `{method_name}()` method must be an int"
    elif not isinstance(observation_space, (spaces.Dict, spaces.Tuple)):
        assert isinstance(
            obs, np.ndarray
        ), f"The observation returned by `{method_name}()` method must be a numpy array"

    # Additional checks for numpy arrays, so the error message is clearer (see GH#1399)
    if isinstance(obs, np.ndarray):
        # check obs dimensions, dtype and bounds
        assert observation_space.shape == obs.shape, (
            f"The observation returned by the `{method_name}()` method does not match"
            f" the shape of the given observation space {observation_space}. Expected:"
            f" {observation_space.shape}, actual shape: {obs.shape}"
        )
        assert np.can_cast(obs.dtype, observation_space.dtype), (
            f"The observation returned by the `{method_name}()` method does not match"
            " the data type (cannot cast) of the given observation space"
            f" {observation_space}. Expected: {observation_space.dtype}, actual dtype:"
            f" {obs.dtype}"
        )
        if isinstance(observation_space, spaces.Box):
            lower_bounds, upper_bounds = observation_space.low, observation_space.high
            # Expose all invalid indices at once
            invalid_indices = np.where(
                np.logical_or(obs < lower_bounds, obs > upper_bounds)
            )
            if (obs > upper_bounds).any() or (obs < lower_bounds).any():
                message = (
                    f"The observation returned by the `{method_name}()` method does not"
                    " match the bounds of the given observation space"
                    f" {observation_space}. \n"
                )
                message += f"{len(invalid_indices[0])} invalid indices: \n"

                for index in zip(*invalid_indices):
                    index_str = ",".join(map(str, index))
                    message += (
                        f"Expected: {lower_bounds[index]} <= obs[{index_str}] <="
                        f" {upper_bounds[index]}, actual value: {obs[index]} \n"
                    )

                raise AssertionError(message)

    assert observation_space.contains(obs), (
        f"The observation returned by the `{method_name}()` method "
        f"does not match the given observation space {observation_space}"
    )


def _check_box_obs(observation_space: spaces.Box, key: str = "") -> None:
    """
    Check that the observation space is correctly formatted
    when dealing with a ``Box()`` space. In particular, it checks:
    - that the dimensions are big enough when it is an image, and that the type matches
    - that the observation has an expected shape (warn the user if not)
    """
    if len(observation_space.shape) not in [1, 3]:
        warnings.warn(
            f"Your observation {key} has an unconventional shape (neither an image, nor"
            " a 1D vector). We recommend you to flatten the observation to have only a"
            " 1D vector or use a custom policy to properly process the data."
        )


def _check_returned_values(
    env: gym.Env, observation_space: spaces.Dict, action_space: spaces.Box
) -> None:
    """
    Check the returned values by the env when calling `.reset()` or `.step()` methods.
    """
    # because env inherits from gymnasium.Env, we assume that `reset()` and `step()` methods exists
    reset_returns = env.reset()
    assert isinstance(reset_returns, tuple), "`reset()` must return a tuple (obs, info)"
    assert (
        len(reset_returns) == 2
    ), f"`reset()` must return a tuple of size 2 (obs, info), not {len(reset_returns)}"
    obs, info = reset_returns
    assert isinstance(info, dict), (
        "The second element of the tuple return by `reset()` must be a dictionary not"
        f" {info}"
    )

    assert isinstance(
        obs, dict
    ), "The observation returned by `reset()` must be a dictionary"
    if not obs.keys() == observation_space.spaces.keys():
        raise AssertionError(
            "The observation keys returned by `reset()` must match the observation "
            f"space keys: {obs.keys()} != {observation_space.spaces.keys()}"
        )

    for key in observation_space.spaces.keys():
        try:
            _check_obs(obs[key], observation_space.spaces[key], "reset")
        except AssertionError as e:
            raise AssertionError(f"Error while checking key={key}: " + str(e)) from e

    # Sample a random action
    action = action_space.sample()
    data = env.step(action)

    assert len(data) == 5, (
        "The `step()` method must return five values: obs, reward, terminated,"
        f" truncated, info. Actual: {len(data)} values returned."
    )

    # Unpack
    obs, reward, terminated, truncated, info = data
    assert isinstance(
        obs, dict
    ), "The observation returned by `step()` must be a dictionary"

    if not obs.keys() == observation_space.spaces.keys():
        raise AssertionError(
            "The observation keys returned by `step()` must match the observation "
            f"space keys: {obs.keys()} != {observation_space.spaces.keys()}"
        )

    for key in observation_space.spaces.keys():
        try:
            _check_obs(obs[key], observation_space.spaces[key], "step")
        except AssertionError as e:
            raise AssertionError(f"Error while checking key={key}: " + str(e)) from e

    # We also allow int because the reward will be cast to float
    assert isinstance(
        reward, (float, int)
    ), "The reward returned by `step()` must be a float"
    assert isinstance(terminated, bool), "The `terminated` signal must be a boolean"
    assert isinstance(truncated, bool), "The `truncated` signal must be a boolean"
    assert isinstance(
        info, dict
    ), "The `info` returned by `step()` must be a python dictionary"


def _check_spaces(env: gym.Env) -> None:
    """
    Check that the observation and action spaces are defined and inherit from spaces.Space. For
    envs that follow the goal-conditioned standard (previously, the gym.GoalEnv interface) we check
    the observation space is gymnasium.spaces.Dict
    """
    gym_spaces = "cf. https://gymnasium.farama.org/api/spaces/"

    assert hasattr(
        env, "observation_space"
    ), f"You must specify an observation space ({gym_spaces})"
    assert hasattr(
        env, "action_space"
    ), f"You must specify an action space ({gym_spaces})"

    assert isinstance(env.observation_space, spaces.Dict), (
        "The observation space must be dictionary, inherited from gymnasium.spaces"
        f" ({gym_spaces})"
    )
    assert isinstance(
        env.action_space, spaces.Box
    ), f"The action space must be box, inherited from gymnasium.spaces ({gym_spaces})"
