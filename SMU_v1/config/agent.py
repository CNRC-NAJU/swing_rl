from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class AgentConfig:
    # Embedding dimensions
    node_type_emb_dim: int = 8
    phase_emb_dim: int = 8
    dphase_emb_dim: int = 8
    mass_emb_dim: int = 8
    gamma_emb_dim: int = 8
    power_emb_dim: int = 8
    active_ratio_emb_dim: int = 8
    perturbation_emb_dim: int = 8

    # GNN configuration
    coupling_hidden_dim: int = 8
    node_hidden_dim: int = 64

    # Activation
    activation: Literal["relu", "gelu", "selu", "elu", "sigmoid", "tanh"] = "elu"

    # Prevent overfitting
    bn_momentum: float = 1.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        assert self.activation in ["relu", "gelu", "selu", "elu", "sigmoid", "tanh"]

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)


AGENT_CONFIG = AgentConfig()
