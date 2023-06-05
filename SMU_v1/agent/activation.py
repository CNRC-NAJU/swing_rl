import torch.nn as nn
from config.agent import _ACTIVATION


def get_activation(name: _ACTIVATION) -> nn.Module:
    match name:
        case "relu":
            return nn.ReLU()
        case "gelu":
            return nn.GELU()
        case "selu":
            return nn.SELU()
        case "elu":
            return nn.ELU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
    raise TypeError(f"No such activation: {name}")
