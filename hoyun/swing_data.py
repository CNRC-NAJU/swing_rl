from typing import TypedDict

import numpy as np
import numpy.typing as npt

arr32 = npt.NDArray[np.float32]


class SwingData(TypedDict):
    edge_list: npt.NDArray[np.int64]  # (E, 2), If (0, 1) is in, (1, 0) is not
    coupling: arr32  # (E, ), Coupling constant of each edge
    phase: arr32  # (N, ), phase of each node
    dphase: arr32  # (N,), dphase(velocity) of each node
    power: arr32  # (N, ), power of each node
    gamma: arr32  # (N,), gamma of each node
    mass: arr32  # (N,), mass of each node

if __name__ == "__main__":
    swing_data = SwingData(
        edge_list=np.array([0, 1], dtype=np.int64),
        phase = np.zeros(2, dtype=np.float32),
        dphase = np.zeros(2, dtype=np.float32),
        coupling = np.array([1.0], dtype=np.float32),
        power = np.array([1.0], dtype=np.float32),
        gamma = np.array([1.0], dtype=np.float32),
        mass = np.array([1.0], dtype=np.float32),
    )

    print(swing_data["edge_list"])