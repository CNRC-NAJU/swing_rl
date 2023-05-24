from dataclasses import dataclass
from typing import Any, Literal, Type

import numpy as np
import numpy.typing as npt


@dataclass
class SwingConfig:
    # which solver to solve swing equation
    name: Literal["rk1", "rk2", "rk4"] = "rk4"

    # step size
    _dt: float = 1e-3

    # dtype
    _dtype: Literal[32, 64] = 64

    def __post__init__(self) -> None:
        assert self.name in ["rk1", "rk2", "rk4"]
        assert self._dtype in [32, 64]

    @property
    def dtype(self) -> Type[np.float32] | Type[np.float64]:
        if self._dtype == 32:
            return np.float32
        elif self._dtype == 64:
            return np.float64
        else:
            raise ValueError(f"Unsupported dtype: float{self._dtype}")

    @property
    def dt(self) -> npt.NDArray:
        return np.array(self._dt, dtype=self.dtype)

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)


SWING_CONFIG = SwingConfig()
