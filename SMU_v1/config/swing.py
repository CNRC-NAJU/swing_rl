from dataclasses import dataclass


@dataclass
class SwingConfig:

    # which solver to solve swing equation
    # e.g., RK1: "rk1",  "rk1_original", "rk1_sparse"
    #       RK2: "rk2",  "rk2_original", "rk2_sparse"
    #       RK4: "rk4",  "rk4_original", "rk4_sparse"
    name: str = "rk4"

    # step size
    dt: float = 1e-3

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
