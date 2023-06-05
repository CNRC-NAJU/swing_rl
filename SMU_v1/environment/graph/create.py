import networkx as nx
import numpy as np
from config import GRAPH_CONFIG
from environment.distribution import create_random_numbers

from .ba import get_ba
from .er import get_er
from .rr import get_rr
from .shk import get_shk


def create_graph(rng: int | np.random.Generator | None = None) -> nx.Graph:
    # Random engine
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    distribution = GRAPH_CONFIG.num_nodes_distribution
    num_nodes = create_random_numbers(
        distribution, 1, rng, dtype="int", clip=(distribution.min, None)
    ).item()

    if GRAPH_CONFIG.topology == "shk":
        return get_shk(
            num_nodes,
            GRAPH_CONFIG.shk_p,
            GRAPH_CONFIG.shk_q,
            GRAPH_CONFIG.shk_r,
            GRAPH_CONFIG.shk_s,
            GRAPH_CONFIG.shk_initial,
            rng,
        )
    elif GRAPH_CONFIG.topology == "ba":
        return get_ba(num_nodes, GRAPH_CONFIG.mean_degree, rng)
    elif GRAPH_CONFIG.topology == "er":
        return get_er(num_nodes, GRAPH_CONFIG.mean_degree, rng=rng)
    elif GRAPH_CONFIG.topology == "rr":
        return get_rr(num_nodes, GRAPH_CONFIG.mean_degree, rng=rng)
    raise TypeError(f"No such graph topology: {GRAPH_CONFIG.topology}")
