import numpy as np

Rng = np.random.Generator | int | None


def uniform_integers_with_sum(
    tot: int, num: int, delta: int, rng: Rng = None
) -> list[int]:
    """
    Find list of (num) positive integers with sum of (tot)
    It's distribution is uniform in [tot/num-delta, tot/num+delta]
    """
    # Random engine
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    # Distribution setting
    avg = round(tot, num)
    min_range, max_range = max(1, avg - delta), avg + delta
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
    tot: int, num: int, std: float, rng: Rng = None
) -> list[int]:
    """
    Find list of (num) positive integers with sum of (tot)
    It's distribution is normal with avg=tot/num and with std
    """
    # Random engine
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    # Distribution setting
    avg = round(tot / num)
    if avg < 1:
        raise ValueError("No solution exists with given parameters.")

    # Find random positive integers with normal distribution
    integers: list[int] = np.clip(
        np.round(rng.normal(loc=avg, scale=std, size=num)).astype(np.int64), 1, None
    ).tolist()

    # Adjust integers to make their sum to be tot
    difference = tot - sum(integers)
    while difference != 0:
        idx = rng.integers(num)
        integers[idx] = max(1, integers[idx] + np.sign(difference))
        difference = tot - sum(integers)

    return integers


def distribute_capacity(
    tot_capacity: int,
    num: int,
    distribution_name: str,
    distribution_param: float,
    rng: Rng = None,
) -> list[int]:
    """
    Randomly distribute the total capacity to given number of sources.
    Available distributions: uniform with parameter delta / normal with parameter sigma

    Return list of capacities
    """
    # Random engine
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    if distribution_name == "uniform":
        capacities = uniform_integers_with_sum(
            tot_capacity, num, int(distribution_param), rng
        )
    elif distribution_name == "normal":
        capacities = normal_integers_with_sum(
            tot_capacity, num, distribution_param, rng
        )
    else:
        raise ValueError(f"No such distribution: {distribution_name}")

    return capacities