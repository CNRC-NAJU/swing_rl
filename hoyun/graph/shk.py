import itertools
from dataclasses import dataclass
from typing import Generator, cast, overload

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt


@dataclass
class SHKParams:
    p: float
    q: float
    r: float
    s: float


class SHK:
    def __init__(
        self,
        initial_num_nodes: int,
        final_num_nodes: int,
        params: SHKParams,
        seed: int | None = None,
    ) -> None:
        """
        initial_num_nodes: Number of nodes to be initialized
        final_num_nodes: Number of nodes at final stage of network growth
        p,q,r,s: parameters for generating network
        """
        self.initial_num_nodes = initial_num_nodes
        self.final_num_nodes = final_num_nodes
        self.p = params.p
        self.q = params.q
        self.r = params.r
        self.s = params.s
        self.random_engine = np.random.default_rng(seed)

        # Randomly choose position of all nodes
        self.pos: npt.NDArray[np.float32] = self.random_engine.random(
            size=(self.final_num_nodes, 2), dtype=np.float32
        )

        # Create initial network
        self.graph = self._MST()  # minimum spanning tree from euclidean distance
        self._add_initial_edge()  # Add initial edges based on fr

        # Grow the network
        for new_node in range(self.initial_num_nodes, self.final_num_nodes):
            self._grow(new_node)

        # save position information to graph
        nx.set_node_attributes(
            G=self.graph,
            values={node: self.pos[node] for node in range(self.final_num_nodes)},
            name="pos",
        )

    def plot(self, ax: plt.Axes) -> None:
        ax.axis("off")
        nx.draw_networkx(
            self.graph,
            pos=nx.get_node_attributes(self.graph, "pos"),
            ax=ax,
            node_size=50,
            node_color="k",
            with_labels=False,
        )

    def _MST(self) -> nx.Graph:
        """Minimum Spanning Tree using euclidean distance between node positions"""
        graph = nx.Graph()

        # Store euclidean distance as edge weight
        for i, j in itertools.combinations(range(self.initial_num_nodes), 2):
            dist = self.get_euclidean_dist(self.pos[i], self.pos[j])
            graph.add_edge(i, j, weight=dist)

        # Generate minimum spanning tree using euclidean distance
        return nx.minimum_spanning_tree(graph)

    def _add_initial_edge(self) -> None:
        """Add Initial edges"""
        additional_num_edge = int(
            self.initial_num_nodes * (1.0 - self.s) * (self.p + self.q)
        )
        node_combination = list(
            itertools.combinations(range(self.initial_num_nodes), 2)
        )

        for _ in range(additional_num_edge):
            # Calculate hop distance of current graph
            hop_dist_gen = cast(
                Generator[tuple[int, dict[int, int]], None, None],
                nx.shortest_path_length(self.graph),
            )

            # hop distance generator to ndarray
            hop_dist = np.zeros(
                (self.initial_num_nodes, self.initial_num_nodes), dtype=np.int64
            )
            for node, hop_dist_dict in hop_dist_gen:
                hop_dist_dict = dict(sorted(hop_dist_dict.items()))  # node index sorted
                hop_dist[node] = np.fromiter(hop_dist_dict.values(), dtype=np.int64)

            # Calculate fr values for each combinations of node i,j
            fr_list = [
                self.get_fr(
                    self.r,
                    hop_dist[i, j],
                    self.get_euclidean_dist(self.pos[i], self.pos[j]),
                )
                for i, j in node_combination
            ]

            # Candidates of new edge is sorted from fr_list: max -> min
            for idx in np.argsort(fr_list)[::-1]:
                if self.graph.has_edge(*node_combination[idx]):
                    continue
                node1, node2 = node_combination[idx]
                break
            else:
                # All node combinations has edges: do nothing
                continue

            # Add edge to chosen node1, node2
            self.graph.add_edge(node1, node2)

    def _grow(self, new_node: int) -> None:
        """Grow network"""
        if self.random_engine.random() < self.s and self.graph.number_of_edges():
            self._bridge(new_node)
        else:
            self._steady(new_node)

    def _bridge(self, new_node: int) -> None:
        """Remove random link and add new node to bridge the removed link"""
        while True:
            # Randomly select existing edge
            node1, node2 = self.random_engine.choice(list(self.graph.edges))
            new_position = self.pos[(node1, node2), :].mean(axis=0)

            # If new position is not occupied in self.pos, save it
            if new_position not in self.pos:
                self.pos[new_node] = new_position
                break

        # Remove target edge and create new edges
        self.graph.remove_edge(node1, node2)
        self.graph.add_edge(node1, new_node)
        self.graph.add_edge(node2, new_node)

    def _steady(self, new_node: int) -> None:
        """Add new node and connect new edges to give more redundancy to network"""
        # Find minimal euclidean distance node w.r.t new_node and add edge
        euclidean_dist_to_new_node = self.get_euclidean_dist(
            self.pos[:new_node], self.pos[new_node]
        )
        node = np.argmin(euclidean_dist_to_new_node).item()
        self.graph.add_edge(node, new_node)

        if self.random_engine.random() < self.p:
            # Add edge with node having maximum fr value
            node = self._find_max_fr_node(new_node, network_size=new_node)
            self.graph.add_edge(node, new_node)

        if self.random_engine.random() < self.q and self.graph.number_of_nodes() > 2:
            # Choose random node and add edge with another node having maximum fr value
            node1 = self.random_engine.integers(new_node)
            node2 = self._find_max_fr_node(node1, network_size=new_node)
            self.graph.add_edge(*sorted([node1, node2]))  # Add edge with sorted index

    def _find_max_fr_node(self, target: int, network_size: int) -> int:
        """
        Find node with maximum fr value w.r.t target node
        Args
            target: target node to calculate fr
            network_size: Network size before adding new node
        Return
            index of node that has maximum fr with target in range [0, network_size]
        """
        # Calculate euclidean distance
        euclidean_dist = self.get_euclidean_dist(
            self.pos[: network_size + 1], self.pos[target]
        )
        euclidean_dist[target] = np.inf

        # Calculate hop distance
        hop_dist_dict = cast(
            dict[int, int], nx.shortest_path_length(self.graph, source=target)
        )
        hop_dist_dict = dict(sorted(hop_dist_dict.items()))  # Sort by node index
        hop_dist = np.fromiter(hop_dist_dict.values(), dtype=np.int64)

        # Calculate fr: target itself has value of 0
        fr = self.get_fr(self.r, hop_dist[:-1], euclidean_dist[:-1])
        for neighbor in self.graph.neighbors(target):
            if neighbor == network_size:
                continue
            fr[neighbor] = 0

        # Return index of node with maximum fr
        return np.argmax(fr).item()

    @overload
    @staticmethod
    def get_fr(r: float, hop_dist: int, euclidean_dist: float) -> float:
        ...

    @overload
    @staticmethod
    def get_fr(
        r: float,
        hop_dist: npt.NDArray[np.int64],
        euclidean_dist: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        ...

    @staticmethod
    def get_fr(r, hop_dist, euclidean_dist):
        """Calculate fr defined at eq 13"""
        return np.power(1.0 + hop_dist, r) / euclidean_dist

    @staticmethod
    def get_euclidean_dist(
        pos1: npt.NDArray[np.float32], pos2: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Calculate euclidean distance between two positions
        Last dimension of pos1,pos2 should be x,y(,z) dimension
        """
        return np.sqrt(np.power(pos1 - pos2, 2.0).sum(axis=-1))


def get_shk(
    num_nodes: int,
    p: float = 0.2,
    q: float = 0.3,
    r: float = 1 / 3,
    s: float = 0.1,
    init_num_nodes: int = 1,
    seed: int | None = None,
) -> nx.Graph:
    grid = SHK(init_num_nodes, num_nodes, SHKParams(p, q, r, s), seed)
    return grid.graph
