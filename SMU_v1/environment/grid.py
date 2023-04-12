import math
from typing import cast

import networkx as nx
import numpy as np
import numpy.typing as npt
from config import GRID_CONFIG, SWING_CONFIG

from .distribution import distribute_capacity
from .graph.create import create_graph
from .graph.utils import (directed2undirected, get_edge_list,
                          get_weighted_adjacency_matrix, repeat_weight)
from .node import (Consumer, ControllableConsumer, Generator, Node, NodeType,
                   Renewable)

Rng = np.random.Generator | int | None
DTYPE = SWING_CONFIG.dtype


class Grid:
    def __init__(
        self,
        graph: nx.Graph | None = None,
        couplings: npt.NDArray[DTYPE] | None = None,
        node_types: list[NodeType] | None = None,
        nodes: list[Node] | None = None,
        rng: Rng = None,
    ) -> None:
        """
        graph: underlying graph structure. This will be constant.
        couplings: coupling strength for each edges on graph.
                  If not given, randomly create couplings with proper distribtution
        node_types: list of node types
        nodes: list of nodes Generator/Renewable/Consumer
               If not given, randomly create nodes with proper distribution
               If node_types is given, nodes should follow it's type
        """
        # Random engine
        if isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)

        # Initialize graph
        self.graph: nx.Graph
        self.edge_list: npt.NDArray[np.int64]
        if graph is None:
            graph = self.create_graph(self.rng)
        self.set_graph(graph)

        # Initialize coupling constants
        self.weighted_adjacency_matrix: npt.NDArray[DTYPE]
        self.couplings: npt.NDArray[DTYPE]
        if couplings is None:
            couplings = self.create_couplings(self.num_edges, self.rng)
        self.set_couplings(couplings)

        # Initialize node types
        self.node_types: list[NodeType]
        self.is_generator: npt.NDArray[np.bool_]
        self.is_renewable: npt.NDArray[np.bool_]
        self.is_consumer: npt.NDArray[np.bool_]
        self.is_controllable_consumer: npt.NDArray[np.bool_]
        if node_types is None:
            node_types = self.create_node_types(self.num_nodes, self.rng)
        self.set_node_types(node_types)

        # Initialize nodes
        self.nodes: list[Node]
        if nodes is None:
            nodes = self.create_nodes(self.node_types, rng=self.rng)
        self.set_nodes(nodes)

        # Activate
        self.activate()

    def __str__(self) -> str:
        return "\n".join(f"Node {i} - {node}" for i, node in enumerate(self.nodes))

    # ---------------------------------- Graph -----------------------------------
    def set_graph(self, graph: nx.Graph) -> None:
        self.graph = graph
        self.edge_list = directed2undirected(get_edge_list(graph))

    def reset_graph(self) -> None:
        """Reset underlying graph of grid
        Coupling constants and node/node types are reset accordingly"""
        graph = self.create_graph(self.rng)
        self.set_graph(graph)

        # Reset coupling, node types accordingly
        self.reset_coupling()
        self.reset_node_types()  # Nodes will also be reset

        # Activate
        self.activate()

    @staticmethod
    def create_graph(rng: Rng = None) -> nx.Graph:
        """Create graph"""
        # Random engine
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        return create_graph(rng)

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    # ---------------------------------- Coupling -----------------------------------
    def set_couplings(self, couplings: npt.NDArray[DTYPE]) -> None:
        assert len(couplings) == self.num_edges

        self.weighted_adjacency_matrix = get_weighted_adjacency_matrix(
            self.graph, couplings
        )
        self.coupling = repeat_weight(couplings)

    def reset_coupling(self) -> None:
        """Reset coupling constants of existing grid"""
        couplings = self.create_couplings(self.num_edges, self.rng)
        self.set_couplings(couplings)

    @staticmethod
    def create_couplings(num_edges: int, rng: Rng = None) -> npt.NDArray[DTYPE]:
        """Create couplings of each edges"""
        # Random engine
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # Assign coupling constant to each edges
        coupling_distribution = GRID_CONFIG.coupling_distribution
        if coupling_distribution.name == "uniform":
            coupling = rng.uniform(
                low=cast(float, coupling_distribution.min),
                high=cast(float, coupling_distribution.max),
                size=num_edges,
            )
        elif coupling_distribution.name == "normal":
            coupling = rng.normal(
                loc=cast(float, coupling_distribution.avg),
                scale=cast(float, coupling_distribution.std),
                size=num_edges,
            )
            assert coupling_distribution.min is not None
            coupling = np.clip(coupling, a_min=coupling_distribution.min, a_max=None)
        else:
            raise ValueError(f"No such distribution {coupling_distribution.name}")
        return coupling.astype(DTYPE, copy=False)

    # ---------------------------------- Node type -----------------------------------
    def set_node_types(self, node_types: list[NodeType]) -> None:
        assert len(node_types) == self.num_nodes

        self.node_types = node_types
        self.is_generator = np.array(
            [node_type is NodeType.GENERATOR for node_type in node_types]
        )
        self.is_renewable = np.array(
            [node_type is NodeType.RENEWABLE for node_type in node_types]
        )
        self.is_consumer = np.array(
            [node_type is NodeType.CONSUMER for node_type in node_types]
        )
        self.is_controllable_consumer = np.array(
            [node_type is NodeType.CONTROLLABLE_CONSUMER for node_type in node_types]
        )

    def reset_node_types(self) -> None:
        """Reset node types of existing grid.
        Nodes is reset accordingly"""
        node_types = self.create_node_types(self.num_nodes, self.rng)
        self.set_node_types(node_types)

        # Reset nodes accordingly
        self.reset_nodes()

        # activate
        self.activate()

    @staticmethod
    def create_node_types(num_nodes: int, rng: Rng = None) -> list[NodeType]:
        """Create list of node types, following proper configurations"""
        # Random engine
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # Number of each node types, following configuration
        num_generators = round(num_nodes * GRID_CONFIG.generator_num_ratio)
        num_renewables = round(num_nodes * GRID_CONFIG.renewable_num_ratio)
        num_controllable_consumers = round(
            num_nodes * GRID_CONFIG.controllable_consumer_num_ratio
        )
        num_consumers = (
            num_nodes - num_generators - num_renewables - num_controllable_consumers
        )

        # list of node types
        node_types = (
            [NodeType.GENERATOR] * num_generators
            + [NodeType.RENEWABLE] * num_renewables
            + [NodeType.CONSUMER] * num_consumers
            + [NodeType.CONTROLLABLE_CONSUMER] * num_controllable_consumers
        )

        # Shuffle the types
        rng.shuffle(node_types)  # type:ignore

        return node_types

    # ------------------------------ Node configuration -----------------------------
    def set_nodes(self, nodes: list[Node]) -> None:
        assert self.match_type(self.node_types, nodes)
        self.nodes = nodes

    def reset_nodes(self) -> None:
        """Reset node of existing grid"""
        nodes = self.create_nodes(self.node_types, self.rng)
        self.set_nodes(nodes)

        # activate
        self.activate()

    @staticmethod
    def create_nodes(node_types: list[NodeType], rng: Rng = None) -> list[Node]:
        """Create list of nodes, following proper configurations and node types"""
        # Random engine
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # Number of each node types, following configuration
        num_generators = sum(
            node_type is NodeType.GENERATOR for node_type in node_types
        )
        num_renewables = sum(
            node_type is NodeType.RENEWABLE for node_type in node_types
        )
        num_consumers = sum(node_type is NodeType.CONSUMER for node_type in node_types)
        num_controllable_consumers = sum(
            node_type is NodeType.CONTROLLABLE_CONSUMER for node_type in node_types
        )

        # Create consumers
        max_units_distribution = GRID_CONFIG.consumer_max_units_distribution
        if max_units_distribution.name == "uniform":
            consumer_max_units = rng.integers(
                low=int(cast(float, max_units_distribution.min)),
                high=int(cast(float, max_units_distribution.max)),
                endpoint=True,
                size=num_consumers,
            )
        elif max_units_distribution.name == "normal":
            consumer_max_units = rng.normal(
                loc=cast(float, max_units_distribution.avg),
                scale=cast(float, max_units_distribution.std),
                size=num_consumers,
            )
            # Round + Clipping: max_units > 1
            consumer_max_units = np.clip(np.round(consumer_max_units), 1, None)
        else:
            raise ValueError(f"No such distribution {max_units_distribution.name}")
        consumers: list[Node] = [
            Consumer(max_units) for max_units in consumer_max_units
        ]

        # Calculate total capacity of consumers/generators/renewables/controllable consumers
        consumer_tot_capacity = abs(sum(consumer.capacity for consumer in consumers))
        generator_tot_capacity = math.ceil(
            consumer_tot_capacity * GRID_CONFIG.generator_spare
        )
        renewable_tot_capacity = math.ceil(
            generator_tot_capacity / GRID_CONFIG.source_ratio
        )
        controllable_consumer_tot_capacity = math.ceil(
            renewable_tot_capacity * GRID_CONFIG.controllable_consumer_spare
        )

        # Create generators, with distributed capacities
        generator_capacities = distribute_capacity(
            generator_tot_capacity,
            num_generators,
            GRID_CONFIG.generator_capacity_distribution,
            rng,
        )
        generators: list[Node] = [
            Generator.from_capacity(capacity) for capacity in generator_capacities
        ]

        # Create renewables, with distributed capacities, masses
        renewable_capacities = distribute_capacity(
            renewable_tot_capacity,
            num_renewables,
            GRID_CONFIG.renewable_capacity_distribution,
            rng,
        )
        renewable_mass_distribution = GRID_CONFIG.renewable_mass_distribution
        if renewable_mass_distribution.name == "uniform":
            renewable_masses = rng.uniform(
                low=cast(float, renewable_mass_distribution.min),
                high=cast(float, renewable_mass_distribution.max),
                size=num_renewables,
            )
        elif renewable_mass_distribution.name == "normal":
            renewable_masses = rng.normal(
                loc=cast(float, renewable_mass_distribution.avg),
                scale=cast(float, renewable_mass_distribution.std),
                size=num_renewables,
            )
            # Round + Clipping: max_units > 1
            renewable_masses = np.clip(np.round(renewable_masses), 0.0, None)
        else:
            raise NotImplementedError(
                f"No such distribution {renewable_mass_distribution.name}"
            )
        renewables: list[Node] = [
            Renewable.from_capacity(capacity, mass)
            for capacity, mass in zip(renewable_capacities, renewable_masses)
        ]

        # Create controllable consumers
        controllable_consumer_capacities = distribute_capacity(
            controllable_consumer_tot_capacity,
            num_controllable_consumers,
            GRID_CONFIG.controllable_consumer_capacity_distribution,
            rng,
        )
        controllable_consumers: list[Node] = [
            ControllableConsumer.from_capacity(capacity)
            for capacity in controllable_consumer_capacities
        ]

        # concatenate generators, renewables, consumers in order of their types
        nodes: list[Node] = []
        for node_type in node_types:
            if node_type is NodeType.GENERATOR:
                nodes.append(generators.pop())
            elif node_type is NodeType.RENEWABLE:
                nodes.append(renewables.pop())
            elif node_type is NodeType.CONSUMER:
                nodes.append(consumers.pop())
            else:
                nodes.append(controllable_consumers.pop())
        return nodes

    @staticmethod
    def match_type(node_types: list[NodeType], nodes: list[Node]) -> bool:
        return node_types == [node.type for node in nodes]

    # --------------------------- grid parameters -------------------------------
    @property
    def powers(self) -> npt.NDArray[DTYPE]:
        return np.array([node.power for node in self.nodes], dtype=DTYPE)

    @property
    def masses(self) -> npt.NDArray[DTYPE]:
        return np.array([node.mass for node in self.nodes], dtype=DTYPE)

    @property
    def gammas(self) -> npt.NDArray[DTYPE]:
        return np.array([node.gamma for node in self.nodes], dtype=DTYPE)

    @property
    def params(self) -> npt.NDArray[DTYPE]:
        return np.stack((self.powers, self.gammas, self.masses))

    @property
    def active_ratios(self) -> npt.NDArray[np.float32]:
        return np.array([node.ratio for node in self.nodes], dtype=np.float32)

    # --------------------------- Power on entire grid -------------------------------
    @property
    def power_imbalance(self) -> int:
        """Imbalance of powers at the entire grid"""
        return sum(node.power for node in self.nodes)

    def activate(self) -> None:
        """Activate each nodes for proper amount"""
        if GRID_CONFIG.initial_rebalance == "directed":
            rebalance = self.rebalance_directed
        elif GRID_CONFIG.initial_rebalance == "undirected":
            rebalance = self.rebalance_undirected
        else:
            raise ValueError(
                f"No such rebalance strategy: {GRID_CONFIG.initial_rebalance}"
            )

        # Increase active units at each of nodes
        for node in self.nodes:
            num_active_units = int(GRID_CONFIG.initial_active_ratio * node.max_units)
            for _ in range(num_active_units):
                node.increase()

        # Resolve power imbalance
        balanced = False
        while not balanced:
            if GRID_CONFIG.initial_rebalance == "directed":
                weights = np.ones(self.num_nodes, dtype=np.float32)
            else:
                weights = self.rng.choice(
                    np.array([-1.0, 1.0], dtype=np.float32), size=self.num_nodes
                )
            balanced = rebalance(weights, GRID_CONFIG.initial_max_rebalance)

    def rebalance_undirected(
        self, weights: npt.NDArray[np.float32], max_trial: int
    ) -> bool:
        """
        Rebalance total power in grid, by perturbation

        Args
        weights: [N, ], all values at [-1.0, 1.0]
        abs(weights): probability that each node will be selected
        sign(weights): If positive, increase active units. Otherwise, decrease
        max_trial: Maximum number of rebalancing trial

        Return
        When rebalancing is successful before max_trial attempts, return True.
        """
        # No need to do rebalancing
        if self.power_imbalance == 0:
            return True

        # Normalize weight
        weights /= np.sum(np.abs(weights))

        # Rebalancing
        for _ in range(max_trial):
            random_idx = self.rng.choice(self.num_nodes, p=np.abs(weights))
            node, weight = self.nodes[random_idx], weights[random_idx]

            # Increase active units if weight is positive, decrease otherwise
            node.increase() if weight > 0 else node.decrease()

            # Check current imbalance state
            if self.power_imbalance == 0:
                return True

        return False

    def rebalance_directed(
        self, weights: npt.NDArray[np.float32], max_trial: int
    ) -> bool:
        """
        Rebalance total power in grid, by increaing/decresing to reduce imbalance
        If power imbalance is positive, increase consumption or decrease production

        Args
        weights: [N, ], all values at [0.0, 1.0]
        max_trial: Maximum number of rebalancing trial

        Return
        When rebalancing is successful before max_trial attempts, return True.
        """
        assert np.all(weights >= 0)
        # No need to do rebalancing
        power_imbalance = self.power_imbalance
        if power_imbalance == 0:
            return True

        is_consumer = self.is_consumer + self.is_controllable_consumer
        if power_imbalance > 0:
            # Direction to increase consumption/decrease production
            weights[~is_consumer] *= -1.0
        else:
            # Direction to decrease consumption/increase production
            weights[is_consumer] *= -1.0
        # Normalize weight
        weights /= np.sum(np.abs(weights))


        for _ in range(max_trial + 1):
            random_idx = self.rng.choice(self.num_nodes, p=np.abs(weights))
            node, weight = self.nodes[random_idx], weights[random_idx]

            # Increase active units if weight is positive, decrease otherwise
            node.increase() if weight > 0 else node.decrease()

            # Check current imbalance state
            if self.power_imbalance == 0:
                return True

        return False

    # ------------------------------ Perturbation ---------------------------------
    def mark_perturbation(self, num: int) -> npt.NDArray[np.int64]:
        """Mark direction of perturbation of each consumers/renewables
        num: How many nodes to be perturbated
        Return: [N, ] whose value is (-1, 0, 1)
            -1 : node will be decreased
            0 : node will not be perturbated
            1 : node will be increased
        """
        # Randomly select nodes to be perturbated: generator is not perturbated
        candidates = (self.is_consumer + self.is_renewable).astype(np.float32)
        indices = self.rng.choice(
            self.num_nodes,
            size=num,
            replace=False,
            p=candidates / candidates.sum(),
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
