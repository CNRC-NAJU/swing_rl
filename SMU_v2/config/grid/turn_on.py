import warnings
from dataclasses import dataclass
from typing import Any, Literal, get_args

from .rebalance import RebalanceConfig

TURN_ON = Literal["equal", "random", "manual", "minimum", "maximum"]


@dataclass(slots=True)
class TurnOnConfig:
    """
    Configuration of turning on the entire grid

    strategy: which strategy to use
    - equal
        All nodes are activated at the same given ratio
        ratio: Active ratio of all nodes
    - random
        Each node is activated in a randomly
    - manual
        Manually assign number of active units
    - minimum
        Activate only single units for every nodes
    - maximum
        Activate all units for every nodes

    rebalance: rebalance strategy after initial active process \\
               See RebalanceConfig documentation
    """

    _strategy: TURN_ON = "equal"
    ratio: float = 0.5
    rebalance: RebalanceConfig = RebalanceConfig(_strategy="directed", max_trials=100)

    def __post__init__(self) -> None:
        assert self.validate_strategy(self._strategy)

    def from_dict(self, config: dict[str, Any]) -> None:
        self.rebalance.from_dict(config.pop("rebalance"))

        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    # --------------------- strategy -------------------------
    @staticmethod
    def validate_strategy(strategy: TURN_ON) -> bool:
        return strategy in get_args(TURN_ON)

    @property
    def strategy(self) -> TURN_ON:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: TURN_ON) -> None:
        if not self.validate_strategy(strategy):
            warnings.warn("Invalid rebalancing strategy. Ignore", stacklevel=2)
            return
        self._strategy = strategy
