from collections import deque

import cloudpickle
import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch

from .type_aliases import Schedule, VecEnvObs


def get_system_info(print_info: bool = True) -> tuple[dict[str, str], str]:
    """
    Retrieve system and python env info for the current system.

    :param print_info: Whether to print or not those infos
    :return: Dictionary summing up the version for each relevant package
        and a formatted string.
    """
    import platform
    import re

    env_info = {
        # In OS, a regex is used to add a space between a "#" and a number to avoid
        # wrongly linking to another issue on GitHub. Example: turn "#42" to "# 42".
        "OS": re.sub(r"#(\d)", r"# \1", f"{platform.platform()} {platform.version()}"),
        "Python": platform.python_version(),
        "Stable-Baselines3": sb3.__version__,
        "PyTorch": torch.__version__,
        "GPU Enabled": str(torch.cuda.is_available()),
        "Numpy": np.__version__,
        "Cloudpickle": cloudpickle.__version__,
        "Gymnasium": gym.__version__,
    }

    env_info_str = ""
    for key, value in env_info.items():
        env_info_str += f"- {key}: {value}\n"
    if print_info:
        print(env_info_str)
    return env_info, env_info_str


def get_device(device: torch.device | str = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"

    # Force conversion to torch.device
    device = torch.device(device)

    # Cuda not available
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


def obs_as_tensor(obs: VecEnvObs, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Moves the observation to the given device.

    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    return {key: torch.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}


def safe_mean(arr: np.ndarray | list | deque) -> float:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr: Numpy array or list of values
    :return:
    """
    return np.nan if len(arr) == 0 else float(np.mean(arr))

def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true).item()
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred).item() / var_y


def get_schedule_fn(value_schedule: Schedule | float | int) -> Schedule:
    """
    Transform (if needed) learning rate and clip range (for PPO)
    to callable.

    :param value_schedule: Constant value of schedule function
    :return: Schedule function (can return constant value)
    """
    # If the passed schedule is a float, create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        return lambda _ : float(value_schedule)
    else:
        return value_schedule
