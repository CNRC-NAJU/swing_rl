import warnings
from dataclasses import dataclass
from typing import Any, Literal, get_args

REWARD = Literal["slope", "area", "weighted_area", "inverse_time", "constant"]


@dataclass(slots=True)
class RewardConfig:
    """
    Configuration for reward functions.
    For the real implementation, refer 'environment/reward.py'

    name: Which reward to be used
    - slope: slope of the dphase at initial fluctuation
    - area: area under dphases graph over time
    - weighted area: area under dphases graph over time, weighted by time
    - inverse_time: proportional to 1/T where T is time
    - constant: constant value
    """

    _name: REWARD = "area"
    scale: float = 1.0
    threshold: float = 0.0

    def __post__init__(self) -> None:
        assert self.validate_name(self._name)

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    # --------------------- name -------------------------
    @staticmethod
    def validate_name(name: REWARD) -> bool:
        return name in get_args(REWARD)

    @property
    def name(self) -> REWARD:
        return self._name

    @name.setter
    def name(self, name: REWARD) -> None:
        if not self.validate_name(name):
            warnings.warn("Invalid reward name. Ignore", stacklevel=2)
            return
        self._name = name
