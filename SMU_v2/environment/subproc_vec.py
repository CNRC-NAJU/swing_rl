"""
SubprocVecEnv implementation of SB3
- All unneccessary codes of SMUD project are removed
- #! comments are from HY
"""

import multiprocessing as mp
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Iterable, Sequence, Type

import cloudpickle
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


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)


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


def _worker(
    remote: mp.connection.Connection,  # type: ignore
    parent_remote: mp.connection.Connection,  # type: ignore
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    parent_remote.close()
    env = env_fn_wrapper.var()
    reset_info: dict[str, Any] | None = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                # convert to SB3 VecEnv api
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


def _flatten_obs(
    obs: list[VecEnvObs] | tuple[VecEnvObs], space: spaces.Space
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
    assert isinstance(
        obs, (list, tuple)
    ), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(
            space.spaces, OrderedDict
        ), "Dict space must have ordered subspaces"
        assert isinstance(
            obs[0], dict
        ), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])  # type: ignore
    elif isinstance(space, spaces.Tuple):
        assert isinstance(
            obs[0], tuple
        ), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))  # type: ignore[index]
    else:
        return np.stack(obs)  # type: ignore[arg-type]


class SubprocVecEnv:
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(
        self, env_fns: list[Callable[[], gym.Env]], start_method: str | None = None
    ):
        #! Create multiple environments
        self.num_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.num_envs)]
        )
        self.processes = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        #! Extract information from single environment
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

        #! For multi-processing communication
        self.waiting = False
        #! Will be only used once: at the end of the program, close the environment
        self.closed = False

        #! reset of each environment
        # store info returned by the reset method
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        # seeds to be used in the next call to env.reset()
        self._seeds: list[int | None] = [None for _ in range(self.num_envs)]
        # options to be used in the next call to env.reset()
        self._options: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        #! Ignore render_mode
        self.render_mode = None
        self.metadata = {"render_modes": [None for _ in range(self.num_envs)]}

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        #! Step for every environments
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

        #! Get results of step
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)  # type: ignore[assignment]

        #! Merge results
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos  # type: ignore[return-value]

    def reset(self) -> VecEnvObs:
        #! Reset for every environments
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))

        #! Get results of reset
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results)  # type: ignore[assignment]

        # Seeds and options are only used once
        self._seeds = [None for _ in range(self.num_envs)]
        self._options = [{} for _ in range(self.num_envs)]

        #! Merge results
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def _get_target_remotes(self, indices: VecEnvIndices) -> list[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """

        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return [self.remotes[i] for i in indices]

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> list[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> list[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

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
