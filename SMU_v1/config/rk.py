from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class RKConfig:
    # step size
    dt: npt.NDArray[np.float32] = np.array(1e-3, dtype=np.float32)
    name: str = "rk4"

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
