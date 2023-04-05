from typing import cast

import networkx as nx
import numpy as np
from config import GraphConfig

from .ba import get_ba
from .er import get_er
from .rr import get_rr
from .shk import get_shk


def create_graph(rng: int | np.random.Generator | None = None) -> nx.Graph:
    config = GraphConfig()

    # Random engine
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    num_nodes_distribution = config.num_nodes_distribution
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

    if config.topology == "shk":
        return get_shk(
            num_nodes,
            config.shk_p,
            config.shk_q,
            config.shk_r,
            config.shk_s,
            config.shk_initial,
            rng,
        )
    elif config.topology == "ba":
        return get_ba(num_nodes, config.mean_degree, rng)
    elif config.topology == "er":
        return get_er(num_nodes, config.mean_degree, rng=rng)
    elif config.topology == "rr":
        return get_rr(num_nodes, config.mean_degree, rng=rng)
    else:
        raise ValueError(f"No such graph topology: {config.topology}")
