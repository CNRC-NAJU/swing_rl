import torch.nn as nn


def get_dropout_layer(dropout: float = 0.0) -> nn.Module:
    """Return batch normalization with given momentum
    if given momentum is 1.0, return identity layer"""
    if dropout == 0.0:
        return nn.Identity()
    else:
        return nn.Dropout(dropout)
