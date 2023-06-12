import warnings
from dataclasses import dataclass
from typing import Any, Literal, get_args

MONITOR_NAME = Literal["null", "inside", "outside"]
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

    solver_name: which swing solver to use
    dt: time interval for swing solving. Constant over single run
    max_time: Maximum time to solve.

    monitor: Monitor the trajectory of swing equation and do early stop
    - monitor_name: which monitor method to use
    - monitor_threshold: threshold value for monitoring
    """

    # which solver to solve swing equation
    _solver_name: SOLVER_NAME = "rk4"

    # time
    dt: float = 1e-2
    max_time: float = 2.0

    # Trajectory monitor
    _monitor_name: MONITOR_NAME = "null"
    monitor_threshold: float = 1e-4

    def __post__init__(self) -> None:
        assert self.validate_solver_name(self._solver_name)
        assert self.validate_monitor_name(self._monitor_name)

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    # --------------------- Solver Name -------------------------
    @staticmethod
    def validate_solver_name(name: SOLVER_NAME) -> bool:
        return name in get_args(SOLVER_NAME)

    @property
    def solver_name(self) -> SOLVER_NAME:
        return self._solver_name

    @solver_name.setter
    def solver_name(self, name: SOLVER_NAME) -> None:
        if not self.validate_solver_name(name):
            warnings.warn(
                f"Invalid swing solver solver_name: {name}. Ignore", stacklevel=2
            )
            return
        self._solver_name = name

    # --------------------- Monitor Name -------------------------
    @staticmethod
    def validate_monitor_name(name: MONITOR_NAME) -> bool:
        return name in get_args(MONITOR_NAME)

    @property
    def monitor_name(self) -> MONITOR_NAME:
        return self._monitor_name

    @monitor_name.setter
    def monitor_name(self, name: MONITOR_NAME) -> None:
        if not self.validate_monitor_name(name):
            warnings.warn(
                f"Invalid swing monitor solver_name: {name}. Ignore", stacklevel=2
            )
            return
        self._monitor_name = name
