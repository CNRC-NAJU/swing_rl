import networkx as nx

from .utils import filter_gcc


def get_rr(num_nodes: int, mean_degree: float | int, gcc: bool = True) -> nx.Graph:
    """Get giant component of random-regular graph
    num_nodes: number of nodes. returning graph could be smaller
    degree: degree of all nodes
    """
    mean_degree = int(mean_degree)
    if (num_nodes * mean_degree) % 2 == 1:
        num_nodes += 1

    graph = nx.random_regular_graph(mean_degree, num_nodes)
    if gcc:
        graph = filter_gcc(graph)
    return graph
