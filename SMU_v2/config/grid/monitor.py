import warnings
from dataclasses import dataclass
from typing import Literal, get_args

MONITOR_NAME = Literal["null", "inside", "outside"]


@dataclass(slots=True)
class MonitorConfig:
    _name: MONITOR_NAME = "null"
    threshold: float = 0.0

    # --------------------- Name -------------------------
    @staticmethod
    def validate_name(name: MONITOR_NAME) -> bool:
        return name in get_args(MONITOR_NAME)

    @property
    def name(self) -> MONITOR_NAME:
        return self._name

    @name.setter
    def name(self, name: MONITOR_NAME) -> None:
        if not self.validate_name(name):
            warnings.warn(
                f"Invalid swing monitor solver_name: {name}. Ignore", stacklevel=2
            )
            return
        self._name = name
