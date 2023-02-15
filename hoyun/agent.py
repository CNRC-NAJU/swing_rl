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
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        hidden_dim: int = 16,
    ):
        # Store graph structure
        self.edge_index = edge_index  # (2, E)
        self.edge_attr = edge_attr  # (E, 1)
        num_nodes = int(self.edge_index.amax()) + 1
        super().__init__(observation_space, num_nodes * hidden_dim)

        self.conv1 = gnn.ChebConv(
            7, hidden_dim, K=2
        )  # phase, dphase, power, gamma, mass, failed_at_this_step, step
        self.conv2 = gnn.ChebConv(hidden_dim, hidden_dim, K=2)

    def forward(self, observations: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        # Convert SB3-format to GNN-format
        num_batch, num_nodes = observations["phase"].shape
        failed_at_this_step = (
            observations["failed_at_this_step"].reshape(-1, num_nodes, 2).argmax(-1)
        )  # (B, N)
        step = observations["step"].argmax(-1).repeat(1, num_nodes)  # (B, N)
        node_feature = torch.stack(  # (BN, 7)
            (
                observations["phase"].flatten(),  # (BN, )
                observations["dphase"].flatten(),  # (BN, )
                observations["power"].flatten(),  # (BN, )
                observations["gamma"].flatten(),  # (BN, )
                observations["mass"].flatten(),  # (BN, )
                failed_at_this_step.flatten(),  # (BN, )
                step.flatten(),  # (BN, )
            ),
            dim=-1,
        )

        # GNN feature extraction
        node_hidden = F.gelu(
            self.conv1(
                node_feature, edge_index=self.edge_index, edge_weight=self.edge_attr
            )
        )  # (BN, hidden_dim)

        node_hidden = F.gelu(
            self.conv2(
                node_hidden, edge_index=self.edge_index, edge_weight=self.edge_attr
            )
        )  # (BN, hidden_dim)

        # Convert GNN-format to SB3-format
        # (B, feature_dim= N * hidden_dim), value squeezed to (-1, 1)
        return torch.tanh(node_hidden).reshape(num_batch, -1)
