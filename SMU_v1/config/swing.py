import warnings
from dataclasses import dataclass
from typing import Any, Literal, Type, get_args

import numpy as np
import numpy.typing as npt

_NAME = Literal["rk1", "rk2", "rk4"]
_DTYPE = Literal[32, 64]


@dataclass(slots=True)
class SwingConfig:
    # which solver to solve swing equation
    _name: _NAME = "rk4"

    # step size
    _dt: float = 1e-3

    # dtype
    _dtype: _DTYPE = 64

    def __post__init__(self) -> None:
        assert self.validate_name(self._name)
        assert self.validate_dtype(self._dtype)

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    # --------------------- Name -------------------------
    @staticmethod
    def validate_name(name: _NAME) -> bool:
        return name in get_args(_NAME)

    @property
    def name(self) -> _NAME:
        return self._name

    @name.setter
    def name(self, value: _NAME) -> None:
        if not self.validate_name(value):
            warnings.warn(f"Invalid swing solver name: {value}. Ignore", stacklevel=2)
            return
        self._name = value

    # --------------------------- dtype ----------------------------
    @staticmethod
    def validate_dtype(dtype: int) -> bool:
        return dtype in get_args(_DTYPE)

    @property
    def dtype(self) -> Type[np.float32] | Type[np.float64]:
        if self._dtype == 32:
            return np.float32
        else:
            return np.float64

    @dtype.setter
    def dtype(self, value: _DTYPE) -> None:
        if not self.validate_dtype(value):
            warnings.warn(f"Unsupported dtype: float{value}. Ignore", stacklevel=2)
            return
        self._dtype = value

    # ---------------------------- dt -----------------------------
    @property
    def dt(self) -> npt.NDArray:
        return np.array(self._dt, dtype=self.dtype)

    @dt.setter
    def dt(self, value: float) -> None:
        self._dt = value


SWING_CONFIG = SwingConfig()
