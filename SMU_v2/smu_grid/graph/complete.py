import networkx as nx


def get_complete(num_nodes: int) -> nx.Graph:
    """Get complete graph, all pairs of nodes have an edge
    num_nodes: number of nodes
    """
    return nx.generators.complete_graph(num_nodes)  # type:ignore

