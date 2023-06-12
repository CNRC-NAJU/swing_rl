import numpy as np
import numpy.typing as npt

from .node import Node


def equal(nodes: list[Node], active_ratio: float) -> list[Node]:
    """
    Turn on the grid with all nodes activated at the same ratio. \\
    Warning: returning nodes is not power-balanced yet

    Args
    nodes: nodes to be turned on
    active_ratio: Active ratio of all nodes

    Return
    nodes: online nodes
    """
    assert 0.0 < active_ratio <= 1.0

    for node in nodes:
        num_active_units = max(1, round(active_ratio * node.max_units))
        node.set(num_active_units)
    return nodes


def random(nodes: list[Node], rng: np.random.Generator) -> list[Node]:
    """
    Turn on the grid with each node activated randomly \\
    Warning: returning nodes is not power-balanced yet

    Args
    nodes: nodes to be turned on
    rng: random engine

    Return
    nodes: online nodes
    """
    for node in nodes:
        num_active_units = rng.integers(1, node.max_units, endpoint=True)
        node.set(int(num_active_units))
    return nodes


def manual(
    nodes: list[Node], num_active_units: npt.NDArray[np.int64] | list[int]
) -> list[Node]:
    """
    Turn on the grid with manually assigning number of active units \\
    Warning: returning nodes could not be power-balanced yet

    Args
    nodes: to be turned on
    num_active_units: [N, ], number of active units of each nodes, should be larger than 1

    Return
    nodes: online nodes
    """
    assert len(nodes) == len(num_active_units)
    assert all(
        1 <= num_active_unit < node.max_units
        for num_active_unit, node in zip(num_active_units, nodes)
    )

    for node, num_active_unit in zip(nodes, num_active_units):
        node.set(int(num_active_unit))
    return nodes


def minimum(nodes: list[Node]) -> list[Node]:
    """
    Turn on the grid with only single active units for every nodes. \\
    Warning: returning nodes could not be power-balanced yet

    Args
    nodes: to be turned on

    Return
    nodes: online nodes
    """
    for node in nodes:
        node.set(1)
    return nodes


def maximum(nodes: list[Node]) -> list[Node]:
    """
    Turn on the grid with all units activated for every nodes. \\
    Warning: returning nodes could not be power-balanced yet

    Args
    nodes: to be turned on

    Return
    nodes: online nodes
    """
    for node in nodes:
        node.set(node.max_units)
    return nodes
