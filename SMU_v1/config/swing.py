from dataclasses import dataclass
from typing import Type, TypeVar

import numpy as np
import numpy.typing as npt


T = TypeVar("T", np.float16, np.float32, np.float64, np.float128)


@dataclass
class SwingConfig:
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

    @classmethod
    @property
    def dtype(cls) -> Type:
        if cls._dtype == 16:
            return np.float16
        elif cls._dtype == 32:
            return np.float32
        elif cls._dtype == 64:
            return np.float64
        elif cls._dtype == 128:
            return np.float128
        else:
            raise ValueError(f"No such _dtype: np.float{cls._dtype}")

    @classmethod
    @property
    def dt(cls) -> npt.NDArray:
        return np.array(cls._dt, dtype=cls.dtype)
