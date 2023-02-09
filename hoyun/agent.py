from collections import OrderedDict

import gym.spaces as spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GraphExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int ,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        hidden_dim: int = 32,
    ):
        super().__init__(observation_space, features_dim)
        # Store graph structure
        self.edge_index = edge_index  # (2, E)
        self.edge_attr = edge_attr  # (E, 1)

        self.conv1 = gnn.ChebConv(
            6, hidden_dim, K=2
        )  # phase, dphase, power, gamma, mass, failed_at_this_step
        self.conv2 = gnn.ChebConv(hidden_dim, hidden_dim, K=2)
        self.out = nn.Linear(hidden_dim + 1, 1)  # hidden+step

    def forward(self, observations: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        num_nodes = observations["phase"].shape[-1]

        failed_at_this_step = (
            observations["failed_at_this_step"].reshape(1, num_nodes, 2).argmax(-1)
        )
        node_feature = torch.stack(
            [
                observations["phase"],  # (0, 2pi)
                observations["dphase"],  # (-inf, inf)
                observations["power"],  # (-inf, inf)
                observations["gamma"],  # (0, inf)
                observations["mass"],  # (0, inf)
                failed_at_this_step,  # 0 or 1
            ],
            dim=-1,
        ).squeeze()  # (N, 6)
        step = observations["step"].argmax(-1).repeat(num_nodes, 1)  # (N, 1)

        node_hidden = F.gelu(
            self.conv1(
                node_feature, edge_index=self.edge_index, edge_weight=self.edge_attr
            )
        )  # (N, hidden_dim)
        node_hidden = F.gelu(
            self.conv2(
                node_hidden, edge_index=self.edge_index, edge_weight=self.edge_attr
            )
        )  # (N, hidden_dim)

        node_hidden = torch.concat([node_hidden, step], dim=-1)  # (N, hidden_dim+1)

        # x = torch.tanh(self.out(node_hidden)).T
        return torch.tanh(self.out(node_hidden)).T  # (1, N), squeezed to (-1, 1)