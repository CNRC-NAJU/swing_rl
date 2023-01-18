import networkx as nx


def get_ws(num_nodes: int, mean_degree: float, p: float) -> nx.Graph:
    """Get watts-strogatz small world graph"""
    return nx.connected_watts_strogatz_graph(num_nodes, int(mean_degree), p)
