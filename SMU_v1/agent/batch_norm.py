import torch.nn as nn
import torch_geometric.nn as gnn


def get_batch_norm_layer(
    in_channels: int, bn_momentum: float = 1.0
) -> nn.Module:
    """Return batch normalization with given momentum
    if given momentum is 1.0, return identity layer"""
    if bn_momentum == 1.0:
        return nn.Identity()
    else:
        return gnn.BatchNorm(in_channels, momentum=bn_momentum)