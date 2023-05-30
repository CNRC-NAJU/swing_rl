import warnings
from dataclasses import dataclass
from typing import Any, Literal

_ACTIVATION = Literal["relu", "gelu", "selu", "elu", "sigmoid", "tanh"]


@dataclass(slots=True)
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
    _activation: _ACTIVATION = "elu"

    # Prevent overfitting
    _bn_momentum: float = 1.0
    _dropout: float = 0.0

    def __post_init__(self) -> None:
        assert self.validate_activation(self._activation)

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    # --------------------- activation -------------------------
    @staticmethod
    def validate_activation(activation: _ACTIVATION) -> bool:
        return activation in ["relu", "gelu", "selu", "elu", "sigmoid", "tanh"]

    @property
    def activation(self) -> _ACTIVATION:
        return self._activation

    @activation.setter
    def activation(self, value: _ACTIVATION) -> None:
        if not self.validate_activation(value):
            warnings.warn(f"Invalid activation: {value}. Ignore", stacklevel=2)
            return
        self._activation = value

    # --------------------- bn_momentum -------------------------
    def validate_bn_momentum(self, bn_momentum: float) -> bool:
        return 0.0 <= bn_momentum <= 1.0

    @property
    def bn_momentum(self) -> float:
        return self._bn_momentum

    @bn_momentum.setter
    def bn_momentum(self, value: float) -> None:
        if not self.validate_bn_momentum(value):
            warnings.warn(f"Invalid bn_momentum: {value}. Ignore", stacklevel=2)
            return
        self._bn_momentum = value

    # --------------------- dropout -------------------------
    def validate_dropout(self, dropout: float) -> bool:
        return 0.0 <= dropout <= 1.0

    @property
    def dropout(self) -> float:
        return self._dropout

    @dropout.setter
    def dropout(self, value: float) -> None:
        if not self.validate_dropout(value):
            warnings.warn(f"Invalid dropout: {value}. Ignore", stacklevel=2)
            return
        self._dropout = value


AGENT_CONFIG = AgentConfig()
