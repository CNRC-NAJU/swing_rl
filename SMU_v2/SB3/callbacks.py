"""
Callback implementation of SB3
- All unneccessary codes of SMUD project are removed
- #! comments are from HY
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from tqdm.auto import tqdm

from .logger import Logger
from .vec_env import VecEnv


class BaseCallback(ABC):
    """
    Base class for callback.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    # The RL model
    # Type hint as string to avoid circular import
    model: "ppo.PPO"  # type: ignore

    def __init__(self, verbose: int = 0):
        super().__init__()
        # Number of time the callback was called
        self.n_calls = 0

        # n_envs * n times env.step() was called
        self.num_timesteps = 0

        self.verbose = verbose
        self.locals: dict[str, Any] = {}
        self.globals: dict[str, Any] = {}

        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent: BaseCallback | None = None

    @property
    def training_env(self) -> VecEnv:
        training_env = self.model.get_env()
        assert training_env is not None, (
            "`model.get_env()` returned None, you must initialize the model with an"
            " environment to use callbacks"
        )
        return training_env

    @property
    def logger(self) -> Logger:
        return self.model.logger

    # Type hint as string to avoid circular import
    def init_callback(self, model: "ppo.PPO") -> None:  # type: ignore
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model
        self._init_callback()

    def on_training_start(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_

        # Update num_timesteps in case training was done before
        self.num_timesteps = self.model.num_timesteps
        self._on_training_start()

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps

        return self._on_step()

    def on_training_end(self) -> None:
        self._on_training_end()

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def update_locals(self, locals_: dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    @abstractmethod
    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True

    def _init_callback(self) -> None:
        pass

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass

    def _on_rollout_end(self) -> None:
        pass

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        """
        Update the references to the local variables on sub callbacks.

        :param locals_: the local variables during rollout collection
        """
        pass


class CallbackList(BaseCallback):
    """
    Class for chaining callbacks.

    :param callbacks: A list of callbacks that will be called
        sequentially.
    """

    def __init__(self, callbacks: list[BaseCallback]):
        super().__init__()
        assert isinstance(callbacks, list)
        self.callbacks = callbacks

    def _init_callback(self) -> None:
        for callback in self.callbacks:
            callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        for callback in self.callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            # Return False (stop training) if at least one callback returns False
            continue_training = callback.on_step() and continue_training
        return continue_training

    def _on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        for callback in self.callbacks:
            callback.update_locals(locals_)


class ConvertCallback(BaseCallback):
    """
    Convert functional callback (old-style) to object.

    :param callback:
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
        self,
        callback: Callable[[dict[str, Any], dict[str, Any]], bool] | None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.callback = callback

    def _on_step(self) -> bool:
        if self.callback is not None:
            return self.callback(self.locals, self.globals)
        return True


class ProgressBarCallback(BaseCallback):
    """
    Display a progress bar when training SB3 agent
    using tqdm and rich packages.
    """

    pbar: tqdm

    def __init__(self) -> None:
        super().__init__()

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(
            total=self.locals["total_timesteps"] - self.model.num_timesteps
        )

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        # Flush and close progress bar
        self.pbar.refresh()
        self.pbar.close()
