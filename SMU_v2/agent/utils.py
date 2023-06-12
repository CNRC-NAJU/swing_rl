import torch.nn as nn


def count_trainable_param(model: nn.Module) -> int:
    """Return number of trainable parameters of model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)