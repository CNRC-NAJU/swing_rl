import itertools
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .node import Node, NodeType


def power_imbalance(nodes: list[Node]) -> int:
    """Imbalance of powers at the given nodes"""
    return sum(node.power for node in nodes)


class Rebalancer(Protocol):
    def __call__(
        self,
        nodes: list[Node],
        weights: npt.NDArray[np.float32 | np.float64],
        max_trials: int,
        rng: np.random.Generator,
    ) -> bool:
        ...


class UndirectedRebalancer:
    """
    Rebalance total power in grid, by moving nodes in direction of sign of weights.
    Nodes are chosen proportional to their weights

    Args
    nodes: nodes to be rebalanced
    weights: [N, ], all values at [-1.0, 1.0]
        abs(weights): probability that each node will be selected
        sign(weights): If positive, increase active units. Otherwise, decrease
    max_trials: Maximum number of rebalancing trial
    rng: random engine

    Return
    When rebalancing is successful before max_trials attempts, return True.
    """

    def __call__(
        self,
        nodes: list[Node],
        weights: npt.NDArray[np.float32 | np.float64],
        max_trials: int,
        rng: np.random.Generator,
    ) -> bool:
        assert np.all(-1.0 <= weights) and np.all(weights <= 1.0)

        # Normalize weight
        weights /= np.sum(np.abs(weights))

        # Rebalancing
        for _ in range(max_trials):
            random_idx = rng.choice(len(nodes), p=np.abs(weights))

            # Increase node if weight is positive, decrease otherwise
            node, weight = nodes[random_idx], weights[random_idx]
            node.increase() if weight > 0 else node.decrease()

            # Check current imbalance state
            if power_imbalance(nodes) == 0:
                return True

        return False


class DirectedRebalancer:
    """
    Rebalance total power in grid, by moving nodes in direction to reduce imbalance
    Nodes are chosen proportional to their weights

    Args
    nodes: nodes to be rebalanced
    weights: [N, ], all values at [0.0, 1.0]
    max_trials: Maximum number of rebalancing trial
    rng: random engine

    Return
    When rebalancing is successful before max_trials attempts, return True.
    """

    def __call__(
        self,
        nodes: list[Node],
        weights: npt.NDArray[np.float32 | np.float64],
        max_trials: int,
        rng: np.random.Generator,
    ) -> bool:
        assert np.all(0.0 <= weights) and np.all(weights <= 1.0)

        # Mark direction
        is_source = np.array(
            [node.type in [NodeType.GENERATOR, NodeType.RENEWABLE] for node in nodes],
            dtype=np.bool_,
        )
        if power_imbalance(nodes) > 0:
            # Direction to decrease source
            weights[is_source] *= -1.0
        else:
            # Direction to decrease user
            weights[~is_source] *= -1.0

        # Normalize weights
        weights /= np.sum(np.abs(weights))

        # Rebalancing
        for _ in range(max_trials):
            random_idx = rng.choice(len(nodes), p=np.abs(weights))

            # Increase node if weight is positive, decrease otherwise
            node, weight = nodes[random_idx], weights[random_idx]
            node.increase() if weight > 0 else node.decrease()

            # Check current imbalance state
            if power_imbalance(nodes) == 0:
                return True

        return False


class DeterministicRebalancer:
    """
    Rebalance total power in grid, by moving nodes in direction to reduce imbalance
    Nodes are chosen in order of their weights

    Args
    nodes: nodes to be rebalanced
    weights: [N, ], all values at [0.0, 1.0]
    max_trials: Maximum number of rebalancing trial

    Return
    When rebalancing is successful before max_trials attempts, return True.
    """

    def __call__(
        self,
        nodes: list[Node],
        weights: npt.NDArray[np.float32 | np.float64],
        max_trials: int,
        rng: np.random.Generator,
    ) -> bool:
        assert np.all(0.0 <= weights) and np.all(weights <= 1.0)

        # Mark direction
        is_source = np.array(
            [node.type in [NodeType.GENERATOR, NodeType.RENEWABLE] for node in nodes],
            dtype=np.bool_,
        )
        if power_imbalance(nodes) > 0:
            # Direction to decrease source
            weights[is_source] *= -1.0
        else:
            # Direction to decrease user
            weights[~is_source] *= -1.0

        # Sort weight in the order of large -> small
        node_order = np.argsort(np.abs(weights))[::-1]

        for _, node_idx in zip(range(max_trials), itertools.cycle(node_order)):
            # Increase node if weight is positive, decrease otherwise
            node, weight = nodes[node_idx], weights[node_idx]
            node.increase() if weight > 0 else node.decrease()

            # Check current imbalance state
            if power_imbalance(nodes) == 0:
                return True

        return False
