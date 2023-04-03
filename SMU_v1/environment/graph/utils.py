from collections import Counter
from typing import Type, TypeVar, cast

import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
from config import GraphConfig

from .ba import get_ba
from .er import get_er
from .rr import get_rr
from .shk import get_shk

T = TypeVar("T", np.float32, np.float64)
arr32 = npt.NDArray[np.float32]
arr64 = npt.NDArray[np.float64]


def get_graph(rng: int | np.random.Generator | None = None) -> nx.Graph:
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


def directed2undirected(
    edge_list: npt.NDArray[np.int64], device: torch.device | None = None
) -> torch.LongTensor:
    """
    Get directed edge list of shape (E, 2)
    Return undirected edge index of shape (2, 2E), for torch_geometric
    """
    edge_index = np.concatenate([edge_list, edge_list[:, (1, 0)]]).T
    return cast(torch.LongTensor, torch.tensor(edge_index, device=device))


def repeat_weight(
    weights: npt.NDArray[T],
    device: torch.device | None = None,
    dtype: Type | None = None,
) -> torch.Tensor:
    """repeat the weight of shape (E, ) or (E, attr) into (2E, ) or (2E, attr)"""
    weights = np.concatenate((weights, weights))  # (2E, ) or (2E, edge_attr)
    return torch.tensor(weights, device=device, dtype=dtype)


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
    """
    Return edge list of shape (E, 2)\\
    If (0,1) is included in the edge list, (1,0) is not included
    """
    return np.array(graph.edges, dtype=np.int64)


def edge_list_2_adjacency_matrix(
    edge_list: npt.NDArray[np.int64],
    weights: npt.NDArray[T],
    num_nodes: int | None = None,
) -> npt.NDArray[T]:
    """
    edge list: (E, 2). If (0,1) is included in the edge list, (1,0) is not included
    weights: (E, ) or (E, 1)
    """
    if num_nodes is None:
        num_nodes = int(edge_list.max()) + 1
    if weights.ndim == 2:
        assert weights.shape[1] == 1

    weighted_adj_matrix = np.zeros((num_nodes, num_nodes), dtype=weights.dtype)

    for (node1, node2), weight in zip(edge_list, weights.squeeze()):
        weighted_adj_matrix[node1, node2] = weight
        weighted_adj_matrix[node2, node1] = weight
    return weighted_adj_matrix


def get_weighted_adjacency_matrix(
    graph: nx.Graph, weights: npt.NDArray[T] | float | None = None
) -> npt.NDArray[T]:
    """
    Return weighted adjacency matrix of shape [N, N].
    If (i,j) is connected weight w, A[i,j] = A[j,i]=w
    If (i,j) is disconnected, A[i,j] = A[j,i] = 0
    Default dtype is float32
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    if weights is None:
        weights = cast(npt.NDArray[T], np.ones(num_edges, dtype=np.float32))
    elif isinstance(weights, float):
        weights = cast(npt.NDArray[T], weights * np.ones(num_edges, dtype=np.float32))

    return edge_list_2_adjacency_matrix(get_edge_list(graph), weights, num_nodes)


def get_weighted_laplacian_matrix(
    graph: nx.Graph, weights: npt.NDArray[T] | float | None = None
) -> npt.NDArray[T]:
    """
    Return weighted laplacian matrix

    weights: (E, ) or (E, 1) if ndarray, assign for each edge
             float if constant over all edges
             None if 1 over all edges
    Default dtype is np.float32
    """

    weighted_adj_matrix = get_weighted_adjacency_matrix(graph, weights)
    dtype = weighted_adj_matrix.dtype

    degree_matrix = np.diag(np.sum(weighted_adj_matrix, axis=0)).view(dtype)
    return (degree_matrix - weighted_adj_matrix).view(dtype)
