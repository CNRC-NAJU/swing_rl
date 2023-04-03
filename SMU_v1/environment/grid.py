import math
from typing import cast

import networkx as nx
import numpy as np
import numpy.typing as npt
from config import GeneratorConfig, GridConfig, RenewableConfig
from graph.utils import (
    directed2undirected,
    get_edge_list,
    get_weighted_adjacency_matrix,
    repeat_weight,
)

from .distribution import distribute_capacity
from .node import NodeType, Consumer, Generator, Node, Renewable

arr32 = npt.NDArray[np.float32]
Rng = np.random.Generator | int | None


class Grid:
    def __init__(
        self,
        graph: nx.Graph,
        coupling: npt.NDArray[np.float32] | None = None,
        nodes: list[Node] | None = None,
        rng: Rng = None,
    ) -> None:
        """
        graph: underlying graph structure. This will be constant.
        coupling: coupling strength for each edges on graph.
                  If not given, randomly create couplings with proper distribtution
        nodes: list of nodes Generator/Renewable/Consumer
               If not given, randomly create nodes with proper distribution
        """
        # --------------- Random setup ------------
        if isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)

        # --------------- Graph setup ------------
        self.__graph = graph  # underlying graph: This should not change over time
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()
        self.edge_list: npt.NDArray[np.int64] = directed2undirected(
            get_edge_list(graph)
        ).numpy()

        # Configure weighted adjacency matrix
        if coupling is None:
            coupling = self.create_coupling(self.num_edges)
        self.set_coupling(coupling)
        self.coupling: arr32 = repeat_weight(coupling).numpy()

        # --------------- Node setup ------------
        # Node types mask
        self.is_consumer: npt.NDArray[np.bool_]
        self.is_generator: npt.NDArray[np.bool_]
        self.is_renewable: npt.NDArray[np.bool_]

        # Assign nodes
        if nodes is None:
            nodes = self.create_nodes(self.num_nodes, rng=self.rng)
        self.set_nodes(nodes)

        # Power on the grid: activate each nodes
        self.activate()

    # ---------------------------------- Graph -----------------------------------
    @property
    def graph(self) -> nx.Graph:
        return self.__graph

    @graph.setter
    def graph(self, _: nx.Graph) -> None:
        raise ValueError("You can't modify graph of the grid")

    def set_coupling(self, coupling: npt.NDArray[np.float32]) -> None:
        """Change coupling of each edges
        Return weighted adjacency matrix"""
        self.coupling = coupling
        self.weighted_adjacency_matrix = get_weighted_adjacency_matrix(
            self.__graph, coupling
        )

    @staticmethod
    def create_coupling(num_edges: int, rng: Rng = None) -> npt.NDArray[np.float32]:
        """Create couplings of each node"""
        # Random engine
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        coupling_distribution = GridConfig.coupling_distribution
        if coupling_distribution.name == "uniform":
            coupling = rng.uniform(
                low=cast(float, coupling_distribution.min),
                high=cast(float, coupling_distribution.max),
                size=num_edges,
            ).astype(np.float32)
        elif coupling_distribution.name == "normal":
            coupling = rng.normal(
                loc=cast(float, coupling_distribution.avg),
                scale=cast(float, coupling_distribution.std),
                size=num_edges,
            ).astype(np.float32)
            assert coupling_distribution.min is not None
            coupling = np.clip(coupling, a_min=coupling_distribution.min, a_max=None)
        else:
            raise ValueError(f"No such distribution {coupling_distribution.name}")
        return coupling

    # ------------------------------ Node configuration -----------------------------
    @property
    def node_types(self) -> list[NodeType]:
        node_types = np.empty(self.num_nodes, dtype=object)
        node_types[self.is_consumer] = NodeType.CONSUMER
        node_types[self.is_generator] = NodeType.GENERATOR
        node_types[self.is_renewable] = NodeType.RENEWABLE

        return node_types.tolist()

    def set_nodes(self, nodes: list[Node]) -> None:
        assert (
            len(nodes) == self.num_nodes
        ), f"Nodes of length {len(nodes)} not match with N={self.num_nodes}"

        self.nodes = nodes
        self.is_consumer = np.array(
            node.type == NodeType.CONSUMER for node in self.nodes
        )
        self.is_generator = np.array(
            node.type == NodeType.GENERATOR for node in self.nodes
        )
        self.is_renewable = np.array(
            node.type == NodeType.RENEWABLE for node in self.nodes
        )

        self.activate()

    @staticmethod
    def create_nodes(
        num_nodes: int, node_types: list[NodeType] | None = None, rng: Rng = None
    ) -> list[Node]:
        """Create list of nodes, following proper configurations
        If node_types is given, returning nodes will have the types"""
        # Random engine
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # Number of each node types, following configuration
        if node_types is None:
            num_generators = int(num_nodes * GridConfig.generator_num_ratio)
            num_renewables = int(num_nodes * GridConfig.renewable_num_ratio)
            num_consumers = num_nodes - num_generators - num_renewables
        else:
            num_generators = sum(node_type is NodeType.GENERATOR for node_type in node_types)
            num_renewables = sum(node_type is NodeType.RENEWABLE for node_type in node_types)
            num_consumers = sum(node_type is NodeType.CONSUMER for node_type in node_types)
            assert num_generators + num_renewables + num_consumers == num_nodes

        # Create consumers
        consumers: list[Node] = [Consumer.randomly(rng) for _ in range(num_consumers)]

        # Calculate total capacity of consumers/generators/renewables
        consumer_tot_capacity = abs(sum(consumer.capacity for consumer in consumers))
        generator_tot_capacity = math.ceil(
            consumer_tot_capacity * GridConfig.generator_spare
        )
        renewable_tot_capacity = math.ceil(
            generator_tot_capacity / GridConfig.source_ratio
        )

        # Create generators, with distributed capacities
        # Randomly distribute generator capacities
        capacities = distribute_capacity(
            generator_tot_capacity,
            num_generators,
            GeneratorConfig.capacity_distribution_name,
            GeneratorConfig.capacity_distribution_param,
            rng,
        )
        generators: list[Node] = [
            Generator.from_capacity(capacity) for capacity in capacities
        ]

        # Create renewables, with distributed capacities
        capacities = distribute_capacity(
            renewable_tot_capacity,
            num_renewables,
            RenewableConfig.capacity_distribution_name,
            RenewableConfig.capacity_distribution_param,
            rng,
        )
        renewables: list[Node] = [
            Renewable.randomly_from_capacity(capacity, rng) for capacity in capacities
        ]

        # concatenate generators, renewables, consumers
        if node_types is None:
            # Randomly shuffle nodes
            nodes = consumers + generators + renewables
            rng.shuffle(nodes)  # type:ignore
        else:
            # Assign nodes following their type
            nodes: list[Node] = []
            for node_type in node_types:
                if node_type is NodeType.GENERATOR:
                    nodes.append(generators.pop())
                elif node_type is NodeType.RENEWABLE:
                    nodes.append(renewables.pop())
                else:
                    nodes.append(consumers.pop())
        return nodes

    @property
    def powers(self) -> npt.NDArray[np.float32]:
        return np.array([node.power for node in self.nodes], dtype=np.float32)

    @property
    def masses(self) -> npt.NDArray[np.float32]:
        return np.array([node.mass for node in self.nodes], dtype=np.float32)

    @property
    def gammas(self) -> npt.NDArray[np.float32]:
        return np.array([node.gamma for node in self.nodes], dtype=np.float32)

    @property
    def active_ratios(self) -> npt.NDArray[np.float32]:
        return np.array([node.ratio for node in self.nodes], dtype=np.float32)

    @property
    def params(self) -> npt.NDArray[np.float32]:
        return np.stack((self.powers, self.gammas, self.masses))

    # --------------------------- Power on entire grid -------------------------------
    @property
    def power_imbalance(self) -> int:
        """Imbalance of powers at the entire grid"""
        return sum(node.power for node in self.nodes)

    def activate(self) -> None:
        """Activate each nodes for proper amount"""
        # Increase active units at each of nodes
        for node in self.nodes:
            num_active_units = int(GridConfig.initial_active_ratio * node.max_units)
            for _ in range(num_active_units):
                node.increase()

        # Resolve power imbalnce
        weights = self.rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32), size=self.num_nodes
        )
        if GridConfig.initial_rebalance == "directed":
            return self.rebalance_directed(weights)
        elif GridConfig.initial_rebalance == "undirected":
            return self.rebalance_undirected(weights)
        else:
            raise ValueError(
                f"No such rebalance strategy: {GridConfig.initial_rebalance}"
            )

    def rebalance_undirected(self, weights: arr32) -> None:
        """
        Rebalance total power in grid, by perturbation
        abs(weights): probability that each node will be selected
        sign(weights): If positive, increase active units. Otherwise, decrease
        """
        weights /= np.sum(np.abs(weights))  # Normalize weight

        # Rebalancing
        while self.power_imbalance != 0:
            random_idx = self.rng.choice(self.num_nodes, p=np.abs(weights))
            node, weight = self.nodes[random_idx], weights[random_idx]

            # Increase active units if weight is positive, decrease otherwise
            node.increase() if weight > 0 else node.decrease()

    def rebalance_directed(self, weights: arr32) -> None:
        """
        Rebalance total power in grid, by increaing/decresing to reduce imbalance
        abs(weights): probability that each node will be selected
        sign(weights): If positive, increase active units. Otherwise, decrease
        """
        # Zero out weights that is not useful for the power imbalance direction
        # i.e., if imbalance is positive, remove production weight
        if self.power_imbalance > 0:
            # Only leave consumer whose weight > 0, source whose weight < 0
            weights[self.is_consumer] = np.clip(
                weights[self.is_consumer], a_min=0.0, a_max=None
            )
            weights[~self.is_consumer] = np.clip(
                weights[~self.is_consumer], a_min=None, a_max=0.0
            )
        else:
            # Only leave consumer whose weight < 0, source whose weight > 0
            weights[self.is_consumer] = np.clip(
                weights[self.is_consumer], a_min=None, a_max=0.0
            )
            weights[~self.is_consumer] = np.clip(
                weights[~self.is_consumer], a_min=0.0, a_max=None
            )
        weights /= np.sum(np.abs(weights))  # Normalize weight

        while self.power_imbalance != 0:
            random_idx = self.rng.choice(self.num_nodes, p=np.abs(weights))
            node, weight = self.nodes[random_idx], weights[random_idx]

            # Increase active units if weight is positive, decrease otherwise
            node.increase() if weight > 0 else node.decrease()

    # ------------------------------ Perturbation ---------------------------------
    def mark_perturbation(self, num: int) -> npt.NDArray[np.int64]:
        """Mark direction of perturbation of each nodes
        num: How many nodes to be perturbated
        Return: [N, ] whose value is (-1, 0, 1)
            -1 : node will be decreased
            0 : node will not be perturbated
            1 : node will be increased
        """
        # Randomly select nodes to be perturbated: generator is not perturbated
        indices = self.rng.choice(
            self.num_nodes, size=num, replace=False, p=(~self.is_generator)
        )

        # Set the direction of perturbation of selected nodes
        perturbation = np.zeros(self.num_nodes, dtype=np.int64)
        for idx in indices:
            node = self.nodes[idx]
            if node.full_active:
                perturbation[idx] = -1
            elif node.full_inactive:
                perturbation[idx] = 1
            else:
                perturbation[idx] = -1 if self.rng.random() < 0.5 else 1

        return perturbation

    def perturbate(self, perturbation: npt.NDArray[np.int64]) -> None:
        for node, direction in zip(self.nodes, perturbation):
            if direction == 0:
                continue
            elif direction == -1:
                node.decrease()
            else:
                node.increase()
