import networkx as nx
import numpy as np


def get_ws(
    num_nodes: int,
    mean_degree: float,
    p: float,
    rng: np.random.Generator | int | None = None,
) -> nx.Graph:
    """Get watts-strogatz small world graph"""
    return nx.connected_watts_strogatz_graph(num_nodes, int(mean_degree), p, seed=rng)
