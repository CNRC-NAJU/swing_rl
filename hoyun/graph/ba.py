import networkx as nx


def get_ba(num_nodes: int, mean_degree: float) -> nx.Graph:
    """Get Barabasi-Albert scale-free network
    num_nodes: number of nodes
    mean_degree: mean degree of resulting graph. should be even
    """
    new_edges = int(mean_degree / 2)
    return nx.barabasi_albert_graph(num_nodes, new_edges)
