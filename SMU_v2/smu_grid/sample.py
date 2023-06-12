from typing import Literal, cast, overload

import numpy as np
import numpy.typing as npt
from config import DistributionConfig


@overload
def sample(
    distribution: DistributionConfig,
    size: int,
    rng: np.random.Generator,
    clip: tuple[int | None, int | None],
    dtype: Literal["int"],
) -> npt.NDArray[np.int64]:
    ...


@overload
def sample(
    distribution: DistributionConfig,
    size: int,
    rng: np.random.Generator,
    clip: tuple[float | None, float | None],
    dtype: Literal["float"],
) -> npt.NDArray[np.float64]:
    ...


def sample(
    distribution: DistributionConfig,
    size: int,
    rng: np.random.Generator,
    clip: tuple[float | None, float | None],
    dtype: Literal["int", "float"],
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
    assert dtype in ["int", "float"]
    if distribution.name == "uniform":
        low = cast(np.float64, distribution.low)
        high = cast(np.float64, distribution.high)

        if dtype == "int":
            result = rng.integers(int(low), int(high), size, endpoint=True)
            clip = tuple(None if c is None else int(c) for c in clip)
        else:
            result = rng.uniform(low, high, size).astype(np.float64, copy=False)

    elif distribution.name == "normal":
        avg = cast(np.float64, distribution.avg)
        std = cast(np.float64, distribution.std)
        result = rng.normal(avg, std, size)

        if dtype == "int":
            result = np.round(result)
            clip = tuple(None if c is None else int(c) for c in clip)
    else:
        raise ValueError(f"Invalid distribution name: {distribution.name}")

    return np.clip(result, a_min=clip[0], a_max=clip[1])


@overload
def sample_restricted_tot(
    distribution: DistributionConfig,
    total: int,
    size: int,
    rng: np.random.Generator,
    clip: tuple[int | None, int | None],
    dtype: Literal["int"],
) -> npt.NDArray[np.int64]:
    ...


@overload
def sample_restricted_tot(
    distribution: DistributionConfig,
    total: float,
    size: int,
    rng: np.random.Generator,
    clip: tuple[float | None, float | None],
    dtype: Literal["float"],
) -> npt.NDArray[np.float64]:
    ...


def sample_restricted_tot(
    distribution: DistributionConfig,
    total,
    size: int,
    rng: np.random.Generator,
    clip,
    dtype: Literal["int", "float"],
) -> npt.NDArray:
    assert dtype in ["int", "float"]
    if distribution.name == "uniform_wo_avg":
        delta = cast(float, distribution.delta)
        if dtype == "int":
            return _sample_uniform_int_restricted_tot(
                total, size, int(delta), clip, rng
            )
        else:
            return _sample_uniform_float_restricted_tot(total, size, delta, clip, rng)

    elif distribution.name == "normal_wo_avg":
        std = cast(float, distribution.std)
        if dtype == "int":
            return _sample_normal_int_restricted_tot(total, size, std, clip, rng)
        else:
            return _sample_normal_float_restricted_tot(total, size, std, clip, rng)
    else:
        raise ValueError(f"Invalid distribution name: {distribution.name}")


def _sample_uniform_int_restricted_tot(
    total: int,
    size: int,
    delta: int,
    clip: tuple[int | None, int | None],
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """
    Find array of integers with sum of total, whose distribution is uniform

    Args
    total: restricted tot. sum(return) = total
    size: number of random numbers
    delta: return value lies in [tot/size - delta, tot/size + delta]
    clip: Range of returning values. Will override delta if given

    Return
    integers: [size, ], sum(integers) = total
    """
    # Low, High range of returning integers
    avg = round(total / size)
    low_range, high_range = avg - delta, avg + delta
    if clip[0] is not None:
        low_range = max(clip[0], low_range)
    if clip[1] is not None:
        high_range = min(clip[1], high_range)
    if total < size * low_range or total > size * high_range:
        raise ValueError("No solution exists with given parameters.")

    # Find random integers with sum is total
    integers = np.zeros(size, dtype=np.int64)
    remaining = total
    for i in range(size):
        # Maximum/minimum range of current integer
        high = min(high_range, remaining - (size - i - 1) * low_range)
        low = max(low_range, remaining - (size - i - 1) * high_range)

        # Randomly select current integer
        value = rng.integers(low=low, high=high, endpoint=True)
        integers[i] = value
        remaining -= value

    rng.shuffle(integers)
    return integers


def _sample_uniform_float_restricted_tot(
    total: float,
    size: int,
    delta: float,
    clip: tuple[float | None, float | None],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """
    Find array of floats with sum of total, whose distribution is uniform

    Args
    total: restricted tot. sum(return) = total
    size: number of random numbers
    delta: return value lies in [tot/size - delta, tot/size + delta]
    clip: Range of returning values. Will override delta range if given

    Return
    floats: [size, ], sum(floats) = total
    """
    # Low, High range of returning floats
    avg = total / size
    low_range, high_range = avg - delta, avg + delta
    if clip[0] is not None:
        low_range = max(clip[0], low_range)
    if clip[1] is not None:
        high_range = min(clip[1], high_range)
    if total < size * low_range or total > size * high_range:
        raise ValueError("No solution exists with given parameters.")

    # Find random integers with sum is total
    floats = np.zeros(size, dtype=np.float64)
    remaining = total
    for i in range(size):
        # Maximum/minimum range of current integer
        high = min(high_range, remaining - (size - i - 1) * low_range)
        low = max(low_range, remaining - (size - i - 1) * high_range)

        # Randomly select current float
        value = rng.uniform(low=low, high=high)
        floats[i] = value
        remaining -= value

    rng.shuffle(floats)
    return floats


def _sample_normal_int_restricted_tot(
    total: int,
    size: int,
    std: float,
    clip: tuple[int | None, int | None],
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """
    Find array of integers with sum of total, whose distribution is normal

    Args
    total: restricted tot. sum(return) = total
    size: number of random numbers
    std: standard deviation of normal distribution
    clip: Range of returning values.

    Return
    integers: [size, ], sum(integers) = total
    """
    # Sample random integers following the distribution
    avg = total / size
    integers: npt.NDArray[np.int64] = np.clip(
        np.round(rng.normal(loc=avg, scale=std, size=size)), *clip, dtype=np.int64
    )

    # Adjust integers to make their sum to be total
    difference = total - sum(integers)
    while difference:
        idx = rng.integers(size)
        integers[idx] = np.clip(integers[idx] + np.sign(difference), *clip)
        difference = total - sum(integers)

    rng.shuffle(integers)
    return integers


def _sample_normal_float_restricted_tot(
    total: float,
    size: int,
    std: float,
    clip: tuple[float | None, float | None],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """
    Find array of floats with sum of total, whose distribution is normal

    Args
    total: restricted tot. sum(return) = total
    size: number of random numbers
    std: standard deviation of normal distribution
    clip: Range of returning values.

    Return
    floats: [size, ], sum(floats) = total
    """
    # Sample random floats following the distribution
    avg = total / size
    floats: npt.NDArray[np.float64] = np.clip(
        rng.normal(loc=avg, scale=std, size=size), *clip, dtype=np.float64
    )

    # Adjust floats to make their sum to be total
    difference = total - sum(floats)
    while difference:
        idx = rng.integers(size)
        floats[idx] = np.clip(floats[idx] + 0.1 * np.sign(difference) * avg, *clip)
        difference = total - sum(floats)

    rng.shuffle(floats)
    return floats
