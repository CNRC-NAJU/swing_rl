from typing import Callable, overload

import numpy as np
import numpy.typing as npt
import torch

arr32 = npt.NDArray[np.float32]
arr64 = npt.NDArray[np.float64]
tensor = torch.Tensor


@overload
def rk1(
    swing_acceleration: Callable[[arr32, arr32], arr32],
    phase: arr32,
    dphase: arr32,
    dt: float,
) -> tuple[arr32, arr32]:
    ...


@overload
def rk1(
    swing_acceleration: Callable[[arr64, arr64], arr64],
    phase: arr64,
    dphase: arr64,
    dt: float,
) -> tuple[arr64, arr64]:
    ...


@overload
def rk1(
    swing_acceleration: Callable[[tensor, tensor], tensor],
    phase: tensor,
    dphase: tensor,
    dt: float,
) -> tuple[tensor, tensor]:
    ...


def rk1(swing_acceleration, phase, dphase, dt):
    acceleration = swing_acceleration(phase, dphase)
    velocity = dphase

    return phase + dt * velocity, dphase + dt * acceleration


@overload
def rk2(
    swing_acceleration: Callable[[arr32, arr32], arr32],
    phase: arr32,
    dphase: arr32,
    dt: float,
) -> tuple[arr32, arr32]:
    ...


@overload
def rk2(
    swing_acceleration: Callable[[arr64, arr64], arr64],
    phase: arr64,
    dphase: arr64,
    dt: float,
) -> tuple[arr64, arr64]:
    ...


@overload
def rk2(
    swing_acceleration: Callable[[tensor, tensor], tensor],
    phase: tensor,
    dphase: tensor,
    dt: float,
) -> tuple[tensor, tensor]:
    ...


def rk2(swing_acceleration, phase, dphase, dt):
    acceleration1 = swing_acceleration(phase, dphase)
    velocity1 = dphase

    temp_phase = phase + dt * velocity1
    temp_dphase = dphase + dt * acceleration1
    acceleration2 = swing_acceleration(temp_phase, temp_dphase)
    velocity2 = temp_dphase

    velocity = 0.5 * (velocity1 + velocity2)
    acceleration = 0.5 * (acceleration1 + acceleration2)

    return phase + dt * velocity, dphase + dt * acceleration


@overload
def rk4(
    swing_acceleration: Callable[[arr32, arr32], arr32],
    phase: arr32,
    dphase: arr32,
    dt: float,
) -> tuple[arr32, arr32]:
    ...


@overload
def rk4(
    swing_acceleration: Callable[[arr64, arr64], arr64],
    phase: arr64,
    dphase: arr64,
    dt: float,
) -> tuple[arr64, arr64]:
    ...


@overload
def rk4(
    swing_acceleration: Callable[[tensor, tensor], tensor],
    phase: tensor,
    dphase: tensor,
    dt: float,
) -> tuple[tensor, tensor]:
    ...


def rk4(swing_acceleration, phase, dphase, dt):
    acceleration1 = swing_acceleration(phase, dphase)
    velocity1 = dphase

    temp_phase = phase + 0.5 * dt * velocity1
    temp_dphase = dphase + 0.5 * dt * acceleration1
    acceleration2 = swing_acceleration(temp_phase, temp_dphase)
    velocity2 = temp_dphase

    temp_phase = phase + 0.5 * dt * velocity2
    temp_dphase = dphase + 0.5 * dt * acceleration2
    acceleration3 = swing_acceleration(temp_phase, temp_dphase)
    velocity3 = temp_dphase

    temp_phase = phase + dt * velocity3
    temp_dphase = dphase + dt * acceleration3
    acceleration4 = swing_acceleration(temp_phase, temp_dphase)
    velocity4 = temp_dphase

    velocity = (velocity1 + 0.5 * velocity2 + 0.5 * velocity3 + velocity4) / 6.0
    acceleration = (
        acceleration1 + 0.5 * acceleration2 + 0.5 * acceleration3 + acceleration4
    ) / 6.0

    return phase + dt * velocity, dphase + dt * acceleration
