import torch.nn as nn


def get_activation(name: str) -> nn.Module:
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

    raise ValueError(f"No such activation: {name}")

