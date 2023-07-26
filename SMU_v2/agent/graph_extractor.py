from collections import OrderedDict

import gymnasium.spaces as spaces
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from SMU_v1.config import observation
from config.rl import OBSERVATION, RL_CONFIG, AgentConfig
from smu_grid.node import NodeType
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .activation import get_activation
from .mlp import MLP


class GraphExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: spaces.Dict, config: AgentConfig | None = None
    ):
        def is_observed(observation: OBSERVATION) -> bool:
            return observation_space[observation].shape != (0,)

        if config is None:
            config = RL_CONFIG.agent

        assert observation_space["perturbation"].shape is not None
        self.num_nodes = observation_space["perturbation"].shape[0]

        super().__init__(observation_space, self.num_nodes * config.node_hidden_dim)

        # ----------------------------- Node embeddings ---------------------------
        self.node_emb_dim = 0
        if is_observed("node_type"):
            self.node_type = nn.Embedding(len(NodeType), config.node_type_emb_dim)
            self.node_emb_dim += config.node_type_emb_dim
        else:
            self.node_type = nn.Identity()

        if is_observed("phase"):
            self.phase = MLP(
                2,
                config.phase_emb_dim,
                config.phase_emb_dim,
                config.activation,
                config.bn_momentum,
                config.dropout,
            )
            self.node_emb_dim += config.phase_emb_dim
        else:
            self.phase = nn.Identity()

        if is_observed("dphase"):
            self.dphase = MLP(
                1,
                config.dphase_emb_dim,
                config.dphase_emb_dim,
                config.activation,
                config.bn_momentum,
                config.dropout,
            )
            self.node_emb_dim += config.dphase_emb_dim
        else:
            self.dphase = nn.Identity()

        if is_observed("mass"):
            self.mass = MLP(
                1,
                config.mass_emb_dim,
                config.mass_emb_dim,
                config.activation,
                config.bn_momentum,
                config.dropout,
            )
            self.node_emb_dim += config.mass_emb_dim
        else:
            self.mass = nn.Identity()

        if is_observed("gamma"):
            self.gamma = MLP(
                1,
                config.gamma_emb_dim,
                config.gamma_emb_dim,
                config.activation,
                config.bn_momentum,
                config.dropout,
            )
            self.node_emb_dim += config.gamma_emb_dim
        else:
            self.gamma = nn.Identity()

        if is_observed("power"):
            self.power = MLP(
                1,
                config.power_emb_dim,
                config.power_emb_dim,
                config.activation,
                config.bn_momentum,
                config.dropout,
            )
            self.node_emb_dim += config.power_emb_dim
        else:
            self.power = nn.Identity()

        if is_observed("active_ratio"):
            self.active_ratio = MLP(
                1,
                config.active_ratio_emb_dim,
                config.active_ratio_emb_dim,
                config.activation,
                config.bn_momentum,
                config.dropout,
            )
            self.node_emb_dim += config.active_ratio_emb_dim
        else:
            self.active_ratio = nn.Identity()

        if is_observed("perturbation"):
            self.perturbation = MLP(
                1,
                config.perturbation_emb_dim,
                config.perturbation_emb_dim,
                config.activation,
                config.bn_momentum,
                config.dropout,
            )
            self.node_emb_dim += config.perturbation_emb_dim
        else:
            self.perturbation = nn.Identity()

        # ----------------------------- Edge embeddings ---------------------------
        self.edge_emb_dim = config.coupling_hidden_dim
        if is_observed("coupling"):
            self.coupling = MLP(
                1,
                config.coupling_hidden_dim,
                1,
                "sigmoid",
                config.bn_momentum,
                config.dropout,
            )
        else:
            self.coupling = nn.Identity()

        # ----------------------------- GNN ---------------------------
        self.conv1 = gnn.GCNConv(self.node_emb_dim, config.node_hidden_dim)
        self.conv2 = gnn.GCNConv(config.node_hidden_dim, config.node_hidden_dim)

        self.act = get_activation(config.activation)

    def encode(
        self, observations: OrderedDict[OBSERVATION, torch.Tensor]
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
                (observations["phase"].cos(), observations["phase"].sin()), dim=-1
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

    def forward(
        self, observations: OrderedDict[OBSERVATION, torch.Tensor]
    ) -> torch.Tensor:
        num_batch, num_nodes = observations["node_type"].shape

        # Different edge_index value for each batch
        edge_index = observations["edge_list"].to(dtype=torch.int64)  # [B, 2, E]
        edge_index += num_nodes * torch.arange(
            num_batch, dtype=torch.int64, device=edge_index.device
        ).reshape(-1, 1, 1)

        # Encode node and edge attributes into higher dimension
        # [B, N, node_emb], [B, E, 1]
        node_attr, edge_attr = self.encode(observations)

        # Convert to pyg format
        node_attr = node_attr.reshape(-1, self.node_emb_dim)  # [BN, node_emb]
        edge_attr = edge_attr.reshape(-1, 1)  # [BE, 1]
        edge_index = edge_index.transpose(0, 1).reshape(2, -1)  # [2, BE]

        # GNN calculation
        x = self.conv1(node_attr, edge_index, edge_weight=edge_attr)  # [BN, node_hid]
        x = self.act(x)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)  # [BN, node_hid]

        # Convert to SB3 format
        return self.act(x).reshape(num_batch, -1)  # [B, N*node_hid]
