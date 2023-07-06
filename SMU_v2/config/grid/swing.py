import warnings
from dataclasses import dataclass
from typing import Any, Literal, get_args

from .monitor import MonitorConfig

SOLVER_NAME = Literal[
    "rk1",
    "rk2",
    "rk4",
    "rk1_original",
    "rk2_original",
    "rk4_original",
    "rk1_sparse",
    "rk2_sparse",
    "rk4_sparse",
]


@dataclass(slots=True)
class SwingConfig:
    """
    Configuration of solving swing equation on grid

    name: which swing solver to use
    dt: time interval for swing solving. Constant over single run
    max_time: Maximum time to solve.

    monitor: Monitor the trajectory of swing equation and do early stop
    - monitor_name: which monitor method to use
    - monitor_threshold: threshold value for monitoring
    """

    # which solver to solve swing equation
    _name: SOLVER_NAME = "rk4"

    # time
    dt: float = 1e-2
    max_time: float = 2.0

    # Trajectory monitor
    monitor: MonitorConfig = MonitorConfig()

    def __post__init__(self) -> None:
        assert self.validate_solver_name(self._name)

    def from_dict(self, config: dict[str, Any]) -> None:
        self.monitor.from_dict(config.pop("monitor"))

        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    # --------------------- Solver Name -------------------------
    @staticmethod
    def validate_solver_name(name: SOLVER_NAME) -> bool:
        return name in get_args(SOLVER_NAME)

    @property
    def name(self) -> SOLVER_NAME:
        return self._name

    @name.setter
    def name(self, name: SOLVER_NAME) -> None:
        if not self.validate_solver_name(name):
            warnings.warn(
                f"Invalid swing solver name: {name}. Ignore", stacklevel=2
            )
            return
        self._name = name
