from collections import Counter
from typing import cast, overload

import networkx as nx
import numpy as np
import numpy.typing as npt

arr32 = npt.NDArray[np.float32]
arr64 = npt.NDArray[np.float64]


def filter_gcc(graph: nx.Graph) -> nx.Graph:
    gcc_nodes = sorted(nx.connected_components(graph), key=len, reverse=True)[0]
    gcc = graph.subgraph(gcc_nodes)
    return nx.convert_node_labels_to_integers(gcc)


def get_mean_degree(graph: nx.Graph) -> float:
    return sum(d for _, d in graph.degree) / graph.number_of_nodes()


def get_degree_distribution(graph: nx.Graph) -> Counter[int]:
    degrees = [d for _, d in graph.degree]
    return Counter(degrees)


def get_edge_list(graph: nx.Graph) -> npt.NDArray[np.int64]:
    """Return edge list of shape [E, 2]
    If (0,1) is included in the edge list, (1,0) is not included"""
    return np.array(graph.edges)


@overload
def edge_list_2_adjacency_matrix(
    edge_list: npt.NDArray[np.int64], weights: arr32, num_nodes: int | None = None
) -> arr32:
    ...


@overload
def edge_list_2_adjacency_matrix(
    edge_list: npt.NDArray[np.int64], weights: arr64, num_nodes: int | None = None
) -> arr64:
    ...


def edge_list_2_adjacency_matrix(
    edge_list: npt.NDArray[np.int64],
    weights: arr32 | arr64,
    num_nodes: int | None = None,
):
    """
    edge list: (E, 2). If (0,1) is included in the edge list, (1,0) is not included
    weights: (E, 1)
    """
    if num_nodes is None:
        num_nodes = cast(int, edge_list.max() + 1)
    weighted_adj_matrix = np.zeros((num_nodes, num_nodes), dtype=weights.dtype)

    for (node1, node2), weight in zip(edge_list, weights):
        weighted_adj_matrix[node1, node2] = weight
        weighted_adj_matrix[node2, node1] = weight
    return weighted_adj_matrix


@overload
def get_weighted_adjacency_matrix(
    graph: nx.Graph, weights: arr32 | float | None
) -> arr32:
    ...


@overload
def get_weighted_adjacency_matrix(graph: nx.Graph, weights: arr64) -> arr64:
    ...


def get_weighted_adjacency_matrix(
    graph: nx.Graph, weights: arr32 | arr64 | float | None = None
):
    """Return weighted adjacency matrix of shape [N, N].
    If (i,j) is connected weight w, A[i,j] = A[j,i]=w
    If (i,j) is disconnected, A[i,j] = A[j,i] = 0
    Default dtype is float32
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    if weights is None:
        weights = np.ones(num_edges, dtype=np.float32)
    elif isinstance(weights, float):
        weights = weights * np.ones(num_edges, dtype=np.float32)

    return edge_list_2_adjacency_matrix(get_edge_list(graph), weights, num_nodes)


@overload
def get_weighted_laplacian_matrix(
    graph: nx.Graph, weights: arr32 | float | None
) -> arr32:
    ...


@overload
def get_weighted_laplacian_matrix(graph: nx.Graph, weights: arr64) -> arr64:
    ...


def get_weighted_laplacian_matrix(
    graph: nx.Graph, weights: arr32 | arr64 | float | None = None
):
    weighted_adj_matrix = get_weighted_adjacency_matrix(graph, weights)
    return np.diag(np.sum(weighted_adj_matrix, axis=0)) - weighted_adj_matrix
