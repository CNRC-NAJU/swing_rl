from collections import OrderedDict
from typing import cast

import gymnasium.spaces as spaces
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from config import AGENT_CONFIG, OBSERVATION_CONFIG
from environment.node import NodeType
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .activation import get_activation
from .mlp import MLP


class GraphExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        self.num_nodes, *_ = cast(tuple[int], observation_space["phase"].shape)
        super().__init__(
            observation_space, self.num_nodes * AGENT_CONFIG.node_hidden_dim
        )

        self.node_emb_dim = 0
        if OBSERVATION_CONFIG.node_type:
            self.node_type = nn.Embedding(len(NodeType), AGENT_CONFIG.node_type_emb_dim)
            self.node_emb_dim += AGENT_CONFIG.node_type_emb_dim
        else:
            self.node_type = nn.Identity()
        if OBSERVATION_CONFIG.phase:
            self.phase = MLP(
                2,
                AGENT_CONFIG.phase_emb_dim,
                AGENT_CONFIG.phase_emb_dim,
                AGENT_CONFIG.activation,
                AGENT_CONFIG.bn_momentum,
                AGENT_CONFIG.dropout,
            )
            self.node_emb_dim += AGENT_CONFIG.phase_emb_dim
        else:
            self.phase = nn.Identity()
        if OBSERVATION_CONFIG.dphase:
            self.dphase = MLP(
                1,
                AGENT_CONFIG.dphase_emb_dim,
                AGENT_CONFIG.dphase_emb_dim,
                AGENT_CONFIG.activation,
                AGENT_CONFIG.bn_momentum,
                AGENT_CONFIG.dropout,
            )
            self.node_emb_dim += AGENT_CONFIG.dphase_emb_dim
        else:
            self.dphase = nn.Identity()
        if OBSERVATION_CONFIG.mass:
            self.mass = MLP(
                1,
                AGENT_CONFIG.mass_emb_dim,
                AGENT_CONFIG.mass_emb_dim,
                AGENT_CONFIG.activation,
                AGENT_CONFIG.bn_momentum,
                AGENT_CONFIG.dropout,
            )
            self.node_emb_dim += AGENT_CONFIG.mass_emb_dim
        else:
            self.mass = nn.Identity()
        if OBSERVATION_CONFIG.gamma:
            self.gamma = MLP(
                1,
                AGENT_CONFIG.gamma_emb_dim,
                AGENT_CONFIG.gamma_emb_dim,
                AGENT_CONFIG.activation,
                AGENT_CONFIG.bn_momentum,
                AGENT_CONFIG.dropout,
            )
            self.node_emb_dim += AGENT_CONFIG.gamma_emb_dim
        else:
            self.gamma = nn.Identity()
        if OBSERVATION_CONFIG.power:
            self.power = MLP(
                1,
                AGENT_CONFIG.power_emb_dim,
                AGENT_CONFIG.power_emb_dim,
                AGENT_CONFIG.activation,
                AGENT_CONFIG.bn_momentum,
                AGENT_CONFIG.dropout,
            )
            self.node_emb_dim += AGENT_CONFIG.power_emb_dim
        else:
            self.power = nn.Identity()
        if OBSERVATION_CONFIG.active_ratio:
            self.active_ratio = MLP(
                1,
                AGENT_CONFIG.active_ratio_emb_dim,
                AGENT_CONFIG.active_ratio_emb_dim,
                AGENT_CONFIG.activation,
                AGENT_CONFIG.bn_momentum,
                AGENT_CONFIG.dropout,
            )
            self.node_emb_dim += AGENT_CONFIG.active_ratio_emb_dim
        else:
            self.active_ratio = nn.Identity()
        if OBSERVATION_CONFIG.perturbation:
            self.perturbation = MLP(
                1,
                AGENT_CONFIG.perturbation_emb_dim,
                AGENT_CONFIG.perturbation_emb_dim,
                AGENT_CONFIG.activation,
                AGENT_CONFIG.bn_momentum,
                AGENT_CONFIG.dropout,
            )
            self.node_emb_dim += AGENT_CONFIG.perturbation_emb_dim
        else:
            self.perturbation = nn.Identity()

        self.edge_emb_dim = AGENT_CONFIG.coupling_hidden_dim
        if OBSERVATION_CONFIG.coupling:
            self.coupling = MLP(
                1,
                AGENT_CONFIG.coupling_hidden_dim,
                1,
                "sigmoid",
                AGENT_CONFIG.bn_momentum,
                AGENT_CONFIG.dropout,
            )
        else:
            self.coupling = nn.Identity()

        self.conv1 = gnn.GCNConv(
            self.node_emb_dim,
            AGENT_CONFIG.node_hidden_dim,
        )
        self.conv2 = gnn.GCNConv(
            AGENT_CONFIG.node_hidden_dim,
            AGENT_CONFIG.node_hidden_dim,
        )

        self.act = get_activation(AGENT_CONFIG.activation)

    def encode(
        self, observations: OrderedDict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        node_type: [B, N] -> [B, N, node_type_emb]
        phase: [B, N] -> [B, N, 2] -> [B, N, phase_emb]
        dphase: [B, N] -> [B, N, 1] -> [B, N, dphase_emb]
        mass: [B, N] -> [B, N, 1] -> [B, N, mass_emb]
        gamma: [B, N] -> [B, N, 1] -> [B, N, gamma_emb]
        power: [B, N] -> [B, N, 1] -> [B, N, power_emb]
        active_ratio: [B, N] -> [B, N, 1] -> [B, N, active_ratio_emb]
        perturbation: [B, N] -> [B, N, 1] -> [B, N, perturbation_emb]

        coupling: [B, E] -> [B, E, 1] -> [B, E, coupling_emb]
        edge_list: [B, 2, 2E]

        Return: tuple of node/edge attribute
        Node attribute: [B, N, node_emb_dim]
        Edge attribute: [B, E, edge_emb_emb]
        """
        node_type_emb = self.node_type(observations["node_type"].to(dtype=torch.int64))
        phase_emb = self.phase(
            torch.stack(
                (observations["phase"].cos(), observations["phase"].sin()),
                dim=-1,
            )
        )
        dphase_emb = self.dphase(observations["dphase"].unsqueeze(-1))
        mass_emb = self.mass(observations["mass"].unsqueeze(-1))
        gamma_emb = self.gamma(observations["gamma"].unsqueeze(-1))
        power_emb = self.power(observations["power"].unsqueeze(-1))
        active_ratio_emb = self.active_ratio(observations["active_ratio"].unsqueeze(-1))
        perturbation_emb = self.perturbation(observations["perturbation"].unsqueeze(-1))

        node_attr = torch.cat(
            (
                node_type_emb,
                phase_emb,
                dphase_emb,
                mass_emb,
                gamma_emb,
                power_emb,
                active_ratio_emb,
                perturbation_emb,
            ),
            dim=-1,
        )

        coupling_emb = self.coupling(observations["coupling"].unsqueeze(-1))
        return node_attr, coupling_emb

    def forward(self, observations: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        edge_index = observations["edge_list"].to(dtype=torch.int64)  # [B, 2, E]
        num_batch = edge_index.shape[0]

        # Encode node and edge attributes into higher dimension
        # [B, N, node_emb], [B, E, 1]
        node_attr, edge_attr = self.encode(observations)

        # Convert to pyg format
        node_attr = node_attr.reshape(-1, self.node_emb_dim)  # [BN, node_emb]
        edge_attr = edge_attr.reshape(-1, 1)  # [BE, 1]
        edge_index = edge_index.transpose(0, 1).reshape(2, -1)  # [2, BE]

        # GNN calculation
        x = self.conv1(node_attr, edge_index, edge_weight=edge_attr)  # [BN, node_hid]
        x = self.conv2(x, edge_index, edge_weight=edge_attr)  # [BN, node_hid]

        # Conver to SB3 format
        return self.act(x).reshape(num_batch, -1)  # [B, N*node_hid]
