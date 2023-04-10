from dataclasses import dataclass
from typing import Any, Type

import numpy as np
import numpy.typing as npt

@dataclass
class SwingConfig():
    # which solver to solve swing equation
    # e.g., RK1: "rk1",  "rk1_original", "rk1_sparse"
    #       RK2: "rk2",  "rk2_original", "rk2_sparse"
    #       RK4: "rk4",  "rk4_original", "rk4_sparse"
    name: str = "rk4"

    # step size
    _dt: float = 1e-3

    # dtype
    _dtype: int = 32

    def __post__init__(self) -> None:
        assert self.name in [
            "rk1",
            "rk2",
            "rk4",
            "rk1_sparse",
            "rk2_sparse",
            "rk4_sparse",
            "rk1_original",
            "rk2_original",
            "rk4_original",
        ]
        assert self._dtype in [16, 32, 64, 128]

    @property
    def dtype(self) -> Type[np.generic]:
        if self._dtype == 16:
            return np.float16
        elif self._dtype == 32:
            return np.float32
        elif self._dtype == 64:
            return np.float64
        elif self._dtype == 128:
            return np.float128
        else:
            raise ValueError(f"No such _dtype: np.float{self._dtype}")

    @property
    def dt(self) -> npt.NDArray:
        return np.array(self._dt, dtype=self.dtype)

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            setattr(self, key, value)


SWING_CONFIG = SwingConfig()