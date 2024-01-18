from typing import Callable, Iterable, NamedTuple

import numpy as np
import torch

# Used when we want to access one or more VecEnv
VecEnvIndices = int | Iterable[int] | None

# VecEnvObs is what is returned by the reset() method
# it contains the observation for each env
VecEnvObs = dict[str, np.ndarray]

# VecEnvStepReturn is what is returned by the step() method
# it contains the observation, reward, done, info for each env
VecEnvStepReturn = tuple[VecEnvObs, np.ndarray, np.ndarray, list[dict]]

# A schedule takes the remaining progress as input
# and ouputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]

TensorDict = dict[str, torch.Tensor]

PyTorchObs = torch.Tensor | TensorDict


class DictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor