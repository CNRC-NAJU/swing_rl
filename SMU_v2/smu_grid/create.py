import math
from typing import cast

import networkx as nx
import numpy as np
import numpy.typing as npt
from config import DistributionConfig
from config.grid import GRID_CONFIG, TOPOLOGY, NumRatioConfig, SHKConfig

from .graph import get_ba, get_complete, get_er, get_rr, get_shk
from .node import Consumer, Generator, Node, NodeType, Renewable, Sink
from .sample import sample, sample_restricted_tot


def create_graph(
    rng: np.random.Generator,
    *,
    num_nodes_distribution: DistributionConfig = GRID_CONFIG.graph.num_nodes_distribution,
    mean_degree_distribution: DistributionConfig = GRID_CONFIG.graph.mean_degree_distribution,
    topology: TOPOLOGY = GRID_CONFIG.graph.topology,
    shk: SHKConfig = GRID_CONFIG.graph.shk,
) -> nx.Graph:
    """
    Create graph

    Args
    rng: random engine
    num_nodes_distribution: Distribution in which num_nodes will be chosen
    mean_degree_distribution: Distribution in which mean degree will be chosen
    topology: which topology to use
    shk: Used only when topology is "shk"
    """

    # Randomly select number of nodes
    distribution = num_nodes_distribution
    minimum = int(cast(float, distribution.low))
    num_nodes = sample(distribution, 1, rng, dtype="int", clip=(minimum, None)).item()

    # Randomly select mean degree
    distribution = mean_degree_distribution
    minimum = cast(float, distribution.low)
    mean_degree = sample(
        distribution, 1, rng, dtype="float", clip=(minimum, None)
    ).item()

    # Create graph according to it's topology
    match topology:
        case "shk":
            return get_shk(num_nodes, shk.p, shk.q, shk.r, shk.s, shk.initial, rng)
        case "ba":
            return get_ba(num_nodes, mean_degree, rng)
        case "er":
            return get_er(num_nodes, mean_degree, rng=rng)
        case "rr":
            return get_rr(num_nodes, mean_degree, rng=rng)
        case "complete":
            return get_complete(num_nodes)
    raise TypeError(f"No such graph topology: {topology}")


def create_couplings(
    num_edges: int,
    rng: np.random.Generator,
    *,
    distribution: DistributionConfig = GRID_CONFIG.coupling_distribution,
) -> npt.NDArray[np.float64]:
    """Create couplings of each edges
    Args
    num_edges: number of edges
    rng: random engine
    distribution: Distribution in which coupling constansts will be chosen

    Return
    coupling constant: [num_edges, ], random couplings following given distribution
    """
    return sample(
        distribution, num_edges, rng, dtype="float", clip=(distribution.low, None)
    )


def create_node_types(
    num_nodes: int,
    rng: np.random.Generator,
    *,
    num_ratio: NumRatioConfig | dict[str, float] = GRID_CONFIG.num_ratio,
) -> list[NodeType]:
    """Create list of node types, following proper number ratios
    Args
    num_nodes: number of nodes
    rng: random engine
    num_ratio: Number ratio of each node types. In order of generator, renewable, consumer, sink

    Return
    node_types: [num_nodes, ], types of each node
    """
    if isinstance(num_ratio, dict):
        assert list(num_ratio.keys()) == ["generator", "renewable", "consumer", "sink"]
        num_ratio = NumRatioConfig(**num_ratio)

    # Number of each node types, following configuration
    num_generators = round(num_nodes * num_ratio.generator)
    num_renewables = round(num_nodes * num_ratio.renewable)
    num_sinks = round(num_nodes * num_ratio.sink)
    num_consumers = num_nodes - num_generators - num_renewables - num_sinks

    # list of node types
    node_types = np.array(
        [NodeType.GENERATOR] * num_generators
        + [NodeType.RENEWABLE] * num_renewables
        + [NodeType.CONSUMER] * num_consumers
        + [NodeType.SINK] * num_sinks,
        dtype=object,
    )

    # Shuffle the types
    rng.shuffle(node_types)
    return node_types.tolist()


def create_nodes(
    node_types: list[NodeType],
    rng: np.random.Generator,
    *,
    consumer_capacity_distribution: DistributionConfig = GRID_CONFIG.consumer_capacity_distribution,
    generator_spare: float = GRID_CONFIG.generator_spare,
    generator_capacity_distribution: DistributionConfig = GRID_CONFIG.generator_capacity_distribution,
    renewable_capacity_distribution: DistributionConfig = GRID_CONFIG.renewable_capacity_distribution,
    renewable_mass_distribution: DistributionConfig = GRID_CONFIG.renewable_mass_distribution,
    sink_capacity_distribution: DistributionConfig = GRID_CONFIG.sink_capacity_distribution,
    source_ratio: float = GRID_CONFIG.source_ratio,
    sink_spare: float = GRID_CONFIG.sink_spare,
) -> list[Node]:
    """Create list of nodes, following proper configurations and given node types
    Args
    node_types: list of node types, where returning node will follow the type
    rng: random engine

    Return
    nodes: [N, ], node object
    """
    # Create consumers
    num_consumers = sum(node_type is NodeType.CONSUMER for node_type in node_types)
    consumers = __create_consumers(num_consumers, consumer_capacity_distribution, rng)
    consumer_tot_capacity = abs(sum(consumer.capacity for consumer in consumers))

    # Create generators, with distributed capacities
    num_generators = sum(node_type is NodeType.GENERATOR for node_type in node_types)
    generator_tot_capacity = math.ceil(consumer_tot_capacity * generator_spare)
    generators = __create_generators(
        num_generators, generator_tot_capacity, generator_capacity_distribution, rng
    )

    # Create renewables, with distributed capacities, masses
    num_renewables = sum(node_type is NodeType.RENEWABLE for node_type in node_types)
    renewable_tot_capacity = math.ceil(generator_tot_capacity * source_ratio)
    renewables = __create_renewables(
        num_renewables,
        renewable_tot_capacity,
        renewable_capacity_distribution,
        renewable_mass_distribution,
        rng,
    )

    # Create sink, with distributed capacities
    num_sinks = sum(node_type is NodeType.SINK for node_type in node_types)
    sink_tot_capacity = math.ceil(renewable_tot_capacity * sink_spare)
    sinks = __create_sinks(
        num_sinks, sink_tot_capacity, sink_capacity_distribution, rng
    )

    # concatenate generators, renewables, consumers in the order of given types
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


def __create_consumers(
    num_consumers: int,
    capacity_distribution: DistributionConfig,
    rng: np.random.Generator,
) -> list[Node]:
    """
    Create a given number of consumers
    Each consumers has random capacity, following given distribution

    Args
    num_consumers: number of consumers
    rng: random engine

    Return: [num_consumers, ], list of consumer nodes
    """
    distribution = capacity_distribution
    minimum = int(cast(float, distribution.low))
    capacities = sample(
        distribution, num_consumers, rng, clip=(minimum, None), dtype="int"
    )
    return [Consumer.from_capacity(capacity) for capacity in capacities]


def __create_generators(
    num_generators: int,
    total_capacity: int,
    capacity_distribution: DistributionConfig,
    rng: np.random.Generator,
) -> list[Node]:
    """
    Create a given number of generators
    Each generators has random capacity, whose sum is given total_capacity

    Args
    num_generators: number of generators
    total_capacity: Sum of capactiy of resulting generators
    rng: random engine

    Return: [num_generators, ], list of generator nodes
    """
    minimum = int(cast(float, capacity_distribution.low))
    capacities = sample_restricted_tot(
        capacity_distribution,
        total_capacity,
        num_generators,
        rng,
        clip=(minimum, None),
        dtype="int",
    )
    return [Generator.from_capacity(capacity) for capacity in capacities]


def __create_renewables(
    num_renewables: int,
    total_capacity: int,
    capacity_distribution: DistributionConfig,
    mass_distribution: DistributionConfig,
    rng: np.random.Generator,
) -> list[Node]:
    """
    Create a given number of renewables
    Each renewables has random capacity, whose sum is given total_capacity
    Mass of nodes follows given distribution

    Args
    num_renewables: number of renewables
    total_capacity: Sum of capactiy of resulting renewables
    rng: random engine

    Return: [num_renewables, ], list of renewable nodes
    """
    # Capacity
    minimum = int(cast(float, capacity_distribution.low))
    capacities = sample_restricted_tot(
        capacity_distribution,
        total_capacity,
        num_renewables,
        rng,
        clip=(minimum, None),
        dtype="int",
    )

    # Mass
    minimum = cast(float, mass_distribution.low)
    masses = sample(
        mass_distribution, num_renewables, rng, clip=(minimum, None), dtype="float"
    )
    return [
        Renewable.from_capacity(capacity, mass)
        for capacity, mass in zip(capacities, masses)
    ]


def __create_sinks(
    num_sinks: int,
    total_capacity: int,
    capacity_distribution: DistributionConfig,
    rng: np.random.Generator,
) -> list[Node]:
    """
    Create a given number of sinks
    Each sinks has random capacity, whose sum is given total_capacity
    Mass of nodes follows given distribution

    Args
    num_sinks: number of sinks
    total_capacity: Sum of capactiy of resulting sinks
    rng: random engine

    Return: [num_sinks, ], list of renewable sink
    """
    minimum = int(cast(float, capacity_distribution.low))
    capacities = sample_restricted_tot(
        capacity_distribution,
        total_capacity,
        num_sinks,
        rng,
        clip=(minimum, None),
        dtype="int",
    )
    return [Sink.from_capacity(capacity) for capacity in capacities]
