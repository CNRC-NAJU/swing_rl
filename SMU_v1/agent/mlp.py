from __future__ import annotations

import torch
import torch.nn as nn

from .activation import get_activation
from .batch_norm import get_batch_norm_layer
from .dropout import get_dropout_layer


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        activation: str,
        bn_momentum: float,
        dropout: float,
        last: bool = False,
    ) -> None:
        """
        Create 2-layer MLP module
        If last, do not use last activations
        """
        super().__init__()
        use_bias = bn_momentum == 1.0
        modules = [
            nn.Linear(in_dim, hidden_dim, bias=use_bias),
            get_batch_norm_layer(hidden_dim, bn_momentum),
            get_activation(activation),
            get_dropout_layer(dropout),
        ]
        if last:
            modules.append(nn.Linear(hidden_dim, out_dim))
        else:
            modules.extend(
                [
                    nn.Linear(hidden_dim, out_dim, bias=use_bias),
                    get_batch_norm_layer(out_dim, bn_momentum),
                    get_activation(activation),
                    get_dropout_layer(dropout),
                ]
            )
        self.mlp = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
