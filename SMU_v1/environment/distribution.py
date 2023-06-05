from typing import Literal, cast, get_args, overload

import numpy as np
import numpy.typing as npt
from config import SWING_CONFIG, DistributionConfig

DTYPE = SWING_CONFIG.dtype


def uniform_integers_with_sum(
    tot: int, num: int, delta: int, low: int, rng: np.random.Generator
) -> list[int]:
    """
    Find list of (num) positive integers with sum of (tot)
    It's distribution is uniform in [tot/num-delta, tot/num+delta]
    """
    # Distribution setting
    avg = round(tot / num)
    min_range, max_range = max(low, avg - delta), avg + delta
    if tot < num * min_range or tot > num * max_range:
        raise ValueError("No solution exists with given parameters.")

    # Find random positive integers with sum of tot
    integers: list[int] = []
    remaining = tot
    for i in range(num):
        # Maximum/minimum range of current integer
        high = min(max_range, remaining - (num - i - 1) * min_range)
        low = max(min_range, remaining - (num - i - 1) * max_range)

        # Randomly select current integer
        integer = rng.integers(low=low, high=high, endpoint=True)
        integers.append(integer)
        remaining -= integer

    rng.shuffle(integers)
    return integers


def normal_integers_with_sum(
    tot: int, num: int, std: float, low: int, rng: np.random.Generator
) -> list[int]:
    """
    Find list of (num) positive integers with sum of (tot)
    It's distribution is normal with avg=tot/num and with std
    """
    # Distribution setting
    avg = tot / num
    if avg < 2:
        raise ValueError("No solution exists with given parameters.")

    # Find random positive integers with normal distribution
    integers: list[int] = np.clip(
        np.round(rng.normal(loc=avg, scale=std, size=num)).astype(np.int64), low, None
    ).tolist()

    # Adjust integers to make their sum to be tot
    difference = tot - sum(integers)
    while difference != 0:
        idx = rng.integers(num)
        integers[idx] = max(low, integers[idx] + np.sign(difference))
        difference = tot - sum(integers)

    rng.shuffle(integers)
    return integers


def distribute_capacity(
    tot_capacity: int,
    num: int,
    distribution: DistributionConfig,
    rng: np.random.Generator,
) -> list[int]:
    """
    Randomly distribute the total capacity to given number of sources.
    Available distributions: uniform with parameter delta / normal with parameter sigma

    Return list of capacities
    """
    if distribution.name == "uniform_wo_avg":
        delta = int(cast(float, distribution.delta))
        low = int(cast(float, distribution.min))
        capacities = uniform_integers_with_sum(tot_capacity, num, delta, low, rng)
    elif distribution.name == "normal_wo_avg":
        std = cast(float, distribution.std)
        low = int(cast(float, distribution.min))
        capacities = normal_integers_with_sum(tot_capacity, num, std, low, rng)
    else:
        raise ValueError(f"Invalid distribution: {distribution.name}")

    return capacities


@overload
def create_random_numbers(
    distribution: DistributionConfig,
    size: int,
    rng: np.random.Generator,
    dtype: Literal["int"],
    clip: tuple[float | None, float | None],
) -> npt.NDArray[np.int64]:
    ...


@overload
def create_random_numbers(
    distribution: DistributionConfig,
    size: int,
    rng: np.random.Generator,
    dtype: Literal["float"],
    clip: tuple[float | None, float | None],
) -> npt.NDArray[DTYPE]:
    ...


def create_random_numbers(
    distribution: DistributionConfig,
    size: int,
    rng: np.random.Generator,
    dtype: Literal["int", "float"],
    clip: tuple[float | None, float | None],
) -> npt.NDArray:
    """
    Create a random numbers

    Args
    distribution: Which distribution the random numbers will follow
    size: number of random numbers
    rng: random number generator
    dtype: integer or float
    clip: minimum, maximum value of returning random values.
            If uniform distribution, clip is prior to the distribution configuration
    """
    assert dtype in get_args(Literal["int", "float"])
    if distribution.name == "uniform":
        low = cast(DTYPE, distribution.min)
        high = cast(DTYPE, distribution.max)

        if dtype == "int":
            result = rng.integers(int(low), int(high), size, endpoint=True)
            clip = tuple(None if c is None else int(c) for c in clip)
        else:
            result = rng.uniform(low, high, size).astype(DTYPE, copy=False)

    elif distribution.name == "normal":
        avg = cast(DTYPE, distribution.avg)
        std = cast(DTYPE, distribution.std)
        result = rng.normal(avg, std, size)

        if dtype == "int":
            result = np.round(result)
            clip = tuple(None if c is None else int(c) for c in clip)
    else:
        raise ValueError(f"Invalid distribution name: {distribution.name}")

    return np.clip(result, a_min=clip[0], a_max=clip[1])
