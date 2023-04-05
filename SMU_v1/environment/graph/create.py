from typing import cast

import networkx as nx
import numpy as np
from config import GraphConfig

from .ba import get_ba
from .er import get_er
from .rr import get_rr
from .shk import get_shk


def create_graph(rng: int | np.random.Generator | None = None) -> nx.Graph:
    # Random engine
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    num_nodes_distribution = GraphConfig.num_nodes_distribution
    if num_nodes_distribution.name == "uniform":
        num_nodes = rng.integers(
            low=int(cast(float, num_nodes_distribution.min)),
            high=int(cast(float, num_nodes_distribution.max)),
            endpoint=True,
        )
    elif num_nodes_distribution.name == "normal":
        num_nodes = np.round(
            rng.normal(
                loc=cast(float, num_nodes_distribution.avg),
                scale=cast(float, num_nodes_distribution.std),
            )
        )
        num_nodes = max(1, num_nodes)  # Clip
    else:
        raise ValueError(f"No such distribution: {num_nodes_distribution.name}")

    if GraphConfig.topology == "shk":
        return get_shk(
            num_nodes,
            GraphConfig.shk_p,
            GraphConfig.shk_q,
            GraphConfig.shk_r,
            GraphConfig.shk_s,
            GraphConfig.shk_initial,
            rng,
        )
    elif GraphConfig.topology == "ba":
        return get_ba(num_nodes, GraphConfig.mean_degree, rng)
    elif GraphConfig.topology == "er":
        return get_er(num_nodes, GraphConfig.mean_degree, rng=rng)
    elif GraphConfig.topology == "rr":
        return get_rr(num_nodes, GraphConfig.mean_degree, rng=rng)
    else:
        raise ValueError(f"No such graph topology: {GraphConfig.topology}")
