import itertools
import math
import warnings

import networkx as nx
import numpy as np
import numpy.typing as npt
from config import GRID_CONFIG, SWING_CONFIG

from .distribution import create_random_numbers, distribute_capacity
from .graph.create import create_graph
from .graph.utils import (directed2undirected, get_edge_list,
                          get_weighted_adjacency_matrix, repeat_weight)
from .node import Consumer, Generator, Node, NodeType, Renewable, Sink

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
        graph: underlying graph structure
        couplings: coupling strength for each edges on graph.
                  If not given, randomly create couplings with proper distribtution
        node_types: list of node types. If not given, randomly create with proper numbers
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
        self._graph: nx.Graph
        self._edge_list: npt.NDArray[np.int64]
        if graph is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.graph = create_graph(self.rng)
        else:
            self.graph = graph

        # Initialize coupling constants
        self._weighted_adjacency_matrix: npt.NDArray[DTYPE]
        self._couplings: npt.NDArray[DTYPE]
        if couplings is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.couplings = self.create_couplings(self.num_edges, self.rng)
        else:
            self.couplings = couplings

        # Initialize node types
        self._node_types: list[NodeType]
        if node_types is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.node_types = self.create_node_types(self.num_nodes, self.rng)
        else:
            self.node_types = node_types

        # Initialize nodes
        self._nodes: list[Node]
        if nodes is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.nodes = self.create_nodes(self._node_types, rng=self.rng)
        else:
            self.nodes = nodes

        # Activate
        self.activate()

    def __str__(self) -> str:
        return "\n".join(f"Node {i} - {node}" for i, node in enumerate(self._nodes))

    # ---------------------------------- Graph -----------------------------------
    @property
    def graph(self) -> nx.Graph:
        return self._graph

    @property
    def edge_list(self) -> npt.NDArray[np.int64]:
        return self._edge_list

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    @graph.setter
    def graph(self, graph: nx.Graph) -> None:
        warnings.warn(
            "Manually assigning graph. Check coupling constant, node type, and node",
            stacklevel=2,
        )
        self._graph = graph
        self._edge_list = directed2undirected(get_edge_list(graph))

    def reset_graph(self) -> None:
        """Reset underlying graph of grid.
        Coupling constants and node/node types are reset accordingly"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.graph = create_graph(self.rng)

        # Reset coupling, node types accordingly
        self.reset_coupling()
        self.reset_node_types()  # Nodes will also be reset

    # ---------------------------------- Coupling -----------------------------------
    @property
    def couplings(self) -> npt.NDArray[DTYPE]:
        """(2E, ). Undirected version"""
        return self._couplings

    @property
    def weighted_adjacency_matrix(self) -> npt.NDArray[DTYPE]:
        """(N, N). weighted by coupling constant"""
        return self._weighted_adjacency_matrix

    @couplings.setter
    def couplings(self, couplings: npt.NDArray[DTYPE]) -> None:
        warnings.warn("Manually assigning coupling constant. Check graph", stacklevel=2)
        assert len(couplings) == self.num_edges

        self._weighted_adjacency_matrix = get_weighted_adjacency_matrix(
            self._graph, couplings  # type: ignore
        )
        self._couplings = repeat_weight(couplings)  # type: ignore

    def reset_coupling(self) -> None:
        """Reset coupling constants of existing grid"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.couplings = self.create_couplings(self.num_edges, self.rng)

    @staticmethod
    def create_couplings(num_edges: int, rng: Rng = None) -> npt.NDArray[DTYPE]:
        """Create couplings of each edges"""
        # Random engine
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # Assign coupling constant to each edges
        distribution = GRID_CONFIG.coupling_distribution

        return create_random_numbers(
            distribution, num_edges, rng, dtype="float", clip=(distribution.min, None)
        )

    # ---------------------------------- Node type -----------------------------------
    @property
    def node_types(self) -> list[NodeType]:
        return self._node_types

    @property
    def is_generator(self) -> npt.NDArray[np.bool_]:
        return np.array(
            [node_type is NodeType.GENERATOR for node_type in self._node_types]
        )

    @property
    def is_renewable(self) -> npt.NDArray[np.bool_]:
        return np.array(
            [node_type is NodeType.RENEWABLE for node_type in self._node_types]
        )

    @property
    def is_consumer(self) -> npt.NDArray[np.bool_]:
        return np.array(
            [node_type is NodeType.CONSUMER for node_type in self._node_types]
        )

    @property
    def is_sink(self) -> npt.NDArray[np.bool_]:
        return np.array([node_type is NodeType.SINK for node_type in self._node_types])

    @node_types.setter
    def node_types(self, node_types: list[NodeType]) -> None:
        warnings.warn("Manually assigning node types. Check graph, node", stacklevel=2)

        assert len(node_types) == self.num_nodes
        self._node_types = node_types

    def reset_node_types(self) -> None:
        """Reset node types of existing grid.
        Nodes is reset accordingly"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.node_types = self.create_node_types(self.num_nodes, self.rng)

        # Reset nodes accordingly
        self.reset_nodes()

    @staticmethod
    def create_node_types(num_nodes: int, rng: Rng = None) -> list[NodeType]:
        """Create list of node types, following proper configurations"""
        # Random engine
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # Number of each node types, following configuration
        num_generators = round(num_nodes * GRID_CONFIG.generator_num_ratio)
        num_renewables = round(num_nodes * GRID_CONFIG.renewable_num_ratio)
        num_sinks = round(num_nodes * GRID_CONFIG.sink_num_ratio)
        num_consumers = num_nodes - num_generators - num_renewables - num_sinks

        # list of node types
        node_types = (
            [NodeType.GENERATOR] * num_generators
            + [NodeType.RENEWABLE] * num_renewables
            + [NodeType.CONSUMER] * num_consumers
            + [NodeType.SINK] * num_sinks
        )

        # Shuffle the types
        rng.shuffle(node_types)  # type:ignore
        return node_types

    # ------------------------------ Node configuration -----------------------------
    @property
    def nodes(self) -> list[Node]:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: list[Node]) -> None:
        warnings.warn("Manually assigning nodes. Check graph, node types", stacklevel=2)
        assert self.match_type(self._node_types, nodes)
        self._nodes = nodes

    def reset_nodes(self) -> None:
        """Reset node of existing grid"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.nodes = self.create_nodes(self._node_types, self.rng)

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
        num_sinks = sum(node_type is NodeType.SINK for node_type in node_types)

        # Create consumers
        distribution = GRID_CONFIG.consumer_capacity_distribution
        consumer_capacities = create_random_numbers(
            distribution, num_consumers, rng, dtype="int", clip=(distribution.min, None)
        )
        consumers: list[Node] = [
            Consumer.from_capacity(capacity) for capacity in consumer_capacities
        ]

        # Calculate total capacity of consumers/generators/renewables/controllable consumers
        consumer_tot_capacity = abs(sum(consumer.capacity for consumer in consumers))
        generator_tot_capacity = math.ceil(
            consumer_tot_capacity * GRID_CONFIG.generator_spare
        )
        renewable_tot_capacity = math.ceil(
            generator_tot_capacity / GRID_CONFIG.source_ratio
        )
        sink_tot_capacity = math.ceil(renewable_tot_capacity * GRID_CONFIG.sink_spare)

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
        distribution = GRID_CONFIG.renewable_mass_distribution
        renewable_masses = create_random_numbers(
            distribution,
            num_renewables,
            rng,
            dtype="float",
            clip=(distribution.min, None),
        )
        renewables: list[Node] = [
            Renewable.from_capacity(capacity, mass)
            for capacity, mass in zip(renewable_capacities, renewable_masses)
        ]

        # Create controllable consumers
        sink_capacities = distribute_capacity(
            sink_tot_capacity,
            num_sinks,
            GRID_CONFIG.sink_capacity_distribution,
            rng,
        )
        sinks: list[Node] = [
            Sink.from_capacity(capacity) for capacity in sink_capacities
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
                nodes.append(sinks.pop())
        return nodes

    @staticmethod
    def match_type(node_types: list[NodeType], nodes: list[Node]) -> bool:
        return node_types == [node.type for node in nodes]

    # --------------------------- grid parameters -------------------------------
    @property
    def powers(self) -> npt.NDArray[DTYPE]:
        return np.array([node.power for node in self._nodes], dtype=DTYPE)

    @property
    def masses(self) -> npt.NDArray[DTYPE]:
        return np.array([node.mass for node in self._nodes], dtype=DTYPE)

    @property
    def gammas(self) -> npt.NDArray[DTYPE]:
        return np.array([node.gamma for node in self._nodes], dtype=DTYPE)

    @property
    def params(self) -> npt.NDArray[DTYPE]:
        return np.stack((self.powers, self.gammas, self.masses))

    @property
    def active_ratios(self) -> npt.NDArray[np.float32]:
        return np.array([node.ratio for node in self._nodes], dtype=np.float32)

    # --------------------------- Power on entire grid -------------------------------
    @property
    def power_imbalance(self) -> int:
        """Imbalance of powers at the entire grid"""
        return sum(node.power for node in self._nodes)

    def activate(self) -> None:
        """Activate each nodes for proper amount
        Sould be called at the very first of grid generation/reset"""

        balanced = False
        while not balanced:
            # Initially activation
            for node in self._nodes:
                # Make only one unit be active
                node.minimize()

                # Turn on some portion of the units
                num_active_units = int(
                    GRID_CONFIG.initial_active_ratio * node.max_units
                )
                for _ in range(num_active_units):
                    node.increase()

            # Resolve power imbalance according to strategy
            if GRID_CONFIG.initial_rebalance == "directed":
                weights = np.ones(self.num_nodes, dtype=np.float32)
                balanced = self.rebalance_directed(
                    weights, GRID_CONFIG.initial_max_rebalance
                )
            elif GRID_CONFIG.initial_rebalance == "undirected":
                weights = self.rng.choice(
                    np.array([-1.0, 1.0], dtype=np.float32), size=self.num_nodes
                )
                balanced = self.rebalance_undirected(
                    weights, GRID_CONFIG.initial_max_rebalance
                )
            else:
                weights = np.ones(self.num_nodes, dtype=np.float32)
                balanced = self.rebalance_deterministic(
                    weights, GRID_CONFIG.initial_max_rebalance
                )

    def rebalance_undirected(
        self, weights: npt.NDArray[np.float32], max_trials: int
    ) -> bool:
        """
        Rebalance total power in grid, by perturbation

        Args
        weights: [N, ], all values at [-1.0, 1.0]
        abs(weights): probability that each node will be selected
        sign(weights): If positive, increase active units. Otherwise, decrease
        max_trials: Maximum number of rebalancing trial

        Return
        When rebalancing is successful before max_trials attempts, return True.
        """
        # No need to do rebalancing
        if self.power_imbalance == 0:
            return True

        # Normalize weight
        weights /= np.sum(np.abs(weights))

        # Rebalancing
        for _ in range(max_trials):
            random_idx = self.rng.choice(self.num_nodes, p=np.abs(weights))

            # Increase node if weight is positive, decrease otherwise
            node, weight = self._nodes[random_idx], weights[random_idx]
            node.increase() if weight > 0 else node.decrease()

            # Check current imbalance state
            if self.power_imbalance == 0:
                return True

        return False

    def rebalance_directed(
        self, weights: npt.NDArray[np.float32], max_trials: int
    ) -> bool:
        """
        Rebalance total power in grid, by increaing/decresing to reduce imbalance
        If power imbalance is positive, increase consumption or decrease production

        Args
        weights: [N, ], all values at [0.0, 1.0]
        max_trials: Maximum number of rebalancing trial

        Return
        When rebalancing is successful before max_trials attempts, return True.
        """
        assert np.all(weights >= 0)
        # No need to do rebalancing
        power_imbalance = self.power_imbalance
        if power_imbalance == 0:
            return True

        is_user = self.is_consumer + self.is_sink
        if power_imbalance > 0:
            # Direction to increase consumption/decrease production
            weights[~is_user] *= -1.0
        else:
            # Direction to decrease consumption/increase production
            weights[is_user] *= -1.0
        # Normalize weight
        weights /= np.sum(np.abs(weights))

        for _ in range(max_trials):
            random_idx = self.rng.choice(self.num_nodes, p=np.abs(weights))

            # Increase node if weight is positive, decrease otherwise
            node, weight = self._nodes[random_idx], weights[random_idx]
            node.increase() if weight > 0 else node.decrease()

            # Check current imbalance state
            if self.power_imbalance == 0:
                return True

        return False

    def rebalance_deterministic(
        self, weights: npt.NDArray[np.float32], max_trials: int
    ) -> bool:
        """
        Rebalance total power in grid, by increaing/decresing to reduce imbalance
        The rebalancing is done in the order of the size of each node weight

        Args
        weights: [N, ], all values at [0.0, 1.0]
        max_trials: Maximum number of rebalancing trial

        Return
        When rebalancing is successful before max_trials attempts, return True.


        각 node들의 weight -> 큰 순서대로 deterministic하게 rebalance

        ex) power 가 모자란 상황
        ordered 되어 있는 순서: [G1, S1, G2, G3, S2]
        G1.increase() fully active
        S1.decrease()
        G2.increase()
        """
        assert np.all(weights >= 0)
        # No need to do rebalancing
        power_imbalance = self.power_imbalance
        if power_imbalance == 0:
            return True

        # Sort weight in the order of large -> small
        node_order = np.argsort(weights)[::-1]

        # Choose the direction to reduce power imbalance
        is_user = self.is_consumer + self.is_sink
        if power_imbalance > 0:
            # Direction to increase consumption/decrease production
            weights[~is_user] *= -1.0
        else:
            # Direction to decrease consumption/increase production
            weights[is_user] *= -1.0

        for _, node_idx in zip(range(max_trials), itertools.cycle(node_order)):
            # Increase node if weight is positive, decrease otherwise
            node, weight = self._nodes[node_idx], weights[node_idx]
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
            node = self._nodes[idx]
            if node.full_active:
                perturbation[idx] = -1
            elif node.full_inactive:
                perturbation[idx] = 1
            else:
                perturbation[idx] = -1 if self.rng.random() < 0.5 else 1

        return perturbation

    def perturbate(self, perturbation: npt.NDArray[np.int64]) -> None:
        """
        Increase/Decrease power of nodes according to perturbation \\
        Othere states such as phase, dphase of nodes are not changing

        Args
        perturbation: return of Grid.mark_perturbation
        """
        for node, direction in zip(self._nodes, perturbation):
            if direction == 0:
                continue
            elif direction == -1:
                node.decrease()
            else:
                node.increase()
