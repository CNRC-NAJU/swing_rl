import warnings
from dataclasses import dataclass
from typing import Any, Literal, get_args

REBALANCE = Literal["directed", "undirected", "deterministic"]


@dataclass(slots=True)
class RebalanceConfig:
    """
    Configuration for (power) rebalance grid

    strategy: Which strategy to be used for rebalancing
    - undirected
        move nodes in direction of sign of weights.
        Nodes are chosen proportional to their weights
    - directed
        move node in direction to reduce imbalance
        Nodes are chosen proportional to their weights
    - deterministic
        move node in direction to reduce imbalance
        Nodes are chosen in order of their weights

    max_trials: Maximum number of trials to try rebalancing
    """

    _strategy: REBALANCE = "directed"
    max_trials: int = 100

    def __post__init__(self) -> None:
        assert self.validate_strategy(self._strategy)

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    # --------------------- strategy -------------------------
    @staticmethod
    def validate_strategy(strategy: REBALANCE) -> bool:
        return strategy in get_args(REBALANCE)

    @property
    def strategy(self) -> REBALANCE:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: REBALANCE) -> None:
        if not self.validate_strategy(strategy):
            warnings.warn("Invalid rebalancing strategy. Ignore", stacklevel=2)
            return
        self._strategy = strategy
