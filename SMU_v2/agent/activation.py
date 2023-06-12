import torch.nn as nn
from config.rl import ACTIVATION


def get_activation(name: ACTIVATION) -> nn.Module:
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
