import warnings
from dataclasses import dataclass
from typing import Any, Literal, get_args

ACTIVATION = Literal["relu", "gelu", "selu", "elu", "sigmoid", "tanh"]


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
    _activation: ACTIVATION = "elu"

    # Prevent overfitting
    _bn_momentum: float = 1.0
    _dropout: float = 0.0

    def __post_init__(self) -> None:
        assert self.validate_activation(self._activation)
        assert self.validate_bn_momentum(self._bn_momentum)
        assert self.validate_dropout(self._dropout)

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    # --------------------- activation -------------------------
    @staticmethod
    def validate_activation(activation: ACTIVATION) -> bool:
        return activation in get_args(ACTIVATION)

    @property
    def activation(self) -> ACTIVATION:
        return self._activation

    @activation.setter
    def activation(self, value: ACTIVATION) -> None:
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
    def bn_momentum(self, bn_momentum: float) -> None:
        if not self.validate_bn_momentum(bn_momentum):
            warnings.warn(f"Invalid bn_momentum: {bn_momentum}. Ignore", stacklevel=2)
            return
        self._bn_momentum = bn_momentum

    # --------------------- dropout -------------------------
    def validate_dropout(self, dropout: float) -> bool:
        return 0.0 <= dropout <= 1.0

    @property
    def dropout(self) -> float:
        return self._dropout

    @dropout.setter
    def dropout(self, dropout: float) -> None:
        if not self.validate_dropout(dropout):
            warnings.warn(f"Invalid dropout: {dropout}. Ignore", stacklevel=2)
            return
        self._dropout = dropout
