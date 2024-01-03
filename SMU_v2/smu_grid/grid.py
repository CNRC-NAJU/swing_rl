import inspect
import warnings
from functools import cache

import networkx as nx
import numpy as np
import numpy.typing as npt
from config import DistributionConfig
from config.grid import (GRID_CONFIG, PerturbationConfig, RebalanceConfig,
                         SwingConfig, TurnOnConfig)
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection

from . import create, rebalance, turn_on
from .graph import utils as gUtils
from .monitor import get_monitor
from .node import Node, NodeType
from .sample import sample_restricted_tot
from .swing.solver import get_swing_solver


class Grid:
    __slots__ = ["_graph", "_couplings", "_nodes", "_rng", "_is_on"]
    """
    graph: underlying graph structure, only including adjacency information \\
    couplings: [E, ] coupling strength for each edges on graph. \\
    nodes: [N, ] list of nodes Generator/Renewable/Consumer/Sink \\
    _is_on: Flag if this grid is offline or online. \\
           When offline, should not run swing equation.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        graph: nx.Graph | None = None,
        couplings: npt.NDArray[np.float64] | None = None,
        node_types: list[NodeType] | None = None,
        nodes: list[Node] | None = None,
        **kwargs,
    ) -> None:
        self._rng = rng

        if graph is None:
            valid_kwargs = inspect.getfullargspec(create.create_graph).kwonlyargs
            graph_kwargs = {
                k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid_kwargs
            }
            graph = create.create_graph(self._rng, **graph_kwargs)

        if couplings is None:
            valid_kwargs = inspect.getfullargspec(create.create_couplings).kwonlyargs
            couplings_kwargs = {
                k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid_kwargs
            }
            couplings = create.create_couplings(
                graph.number_of_edges(), self._rng, **couplings_kwargs
            )

        if node_types is None:
            valid_kwargs = inspect.getfullargspec(create.create_node_types).kwonlyargs
            node_types_kwargs = {
                k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid_kwargs
            }
            node_types = create.create_node_types(
                graph.number_of_nodes(), self._rng, **node_types_kwargs
            )

        if nodes is None:
            valid_kwargs = inspect.getfullargspec(create.create_nodes).kwonlyargs
            nodes_kwargs = {
                k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid_kwargs
            }
            nodes = create.create_nodes(node_types, self._rng, **nodes_kwargs)

        self.validate_graph_couplings(graph, couplings)
        self.validate_graph_node_types(graph, node_types)
        self.validate_node_types_nodes(node_types, nodes)

        self._graph = graph
        self._couplings = couplings
        self._nodes = nodes
        self._is_on = False

    def __str__(self) -> str:
        return "\n".join(f"Node {i} - {node}" for i, node in enumerate(self._nodes))

    # ----------------------- Plot -----------------------
    def draw(self, ax: Axes, cursors: bool = True, **kwargs) -> PathCollection:
        """
        Draw current grid

        Args
        ax: axis to plot
        kwargs: any arguments passed for networkx.draw

        Return
        nodes: list of node object in ax.scatter format
        """

        try:
            pos = nx.function.get_node_attributes(self._graph, "pos")
        except KeyError:
            pos = nx.drawing.layout.spring_layout(self._graph)
        nx.drawing.nx_pylab.draw(self._graph, pos=pos, ax=ax, **kwargs)

        node_object = [
            artist for artist in ax.get_children() if isinstance(artist, PathCollection)
        ][0]

        if cursors:
            try:
                import mplcursors

                cursor = mplcursors.cursor(node_object, hover=2)

                @cursor.connect("add")
                def on_hover(sel: mplcursors.Selection) -> None:
                    sel.annotation.set_text(self._nodes[sel.index])
                    sel.annotation.set_bbox({"facecolor": "w"})

            except ImportError:
                warnings.warn("You should install mplcursors package", stacklevel=2)

        return node_object

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    # ----------------------- Validity of attributes -----------------------
    @staticmethod
    def validate_graph_couplings(
        graph: nx.Graph, couplings: npt.NDArray[np.float64]
    ) -> None:
        """Check if graph and coupling constants are compatible"""
        assert couplings.ndim == 1, f"Error: {couplings.shape=}"
        assert graph.number_of_edges() == len(couplings)

    @staticmethod
    def validate_graph_node_types(graph: nx.Graph, node_types: list[NodeType]) -> None:
        """Check if graph and node types are compatible"""
        assert graph.number_of_nodes() == len(node_types)

    @staticmethod
    def validate_node_types_nodes(
        node_types: list[NodeType], nodes: list[Node]
    ) -> None:
        """Check if node types and nodes are compatible"""
        assert node_types == [node.type for node in nodes]

    # ----------------------- Graph, Adjacency information -----------------------
    @property
    @cache
    def graph(self) -> nx.Graph:
        """Underlying graph structure"""
        return self._graph

    @property
    @cache
    def num_nodes(self) -> int:
        """Number of nodes at the underlying graph"""
        return self._graph.number_of_nodes()

    @property
    @cache
    def num_edges(self) -> int:
        """Number of edges at the underlying graph"""
        return self._graph.number_of_edges()

    @property
    @cache
    def edge_list(self) -> npt.NDArray[np.int64]:
        """[2, E], edge list"""
        return gUtils.get_edge_list(self._graph)

    @property
    @cache
    def undirected_edge_list(self) -> npt.NDArray[np.int64]:
        """[2, 2E], Undirected edge list"""
        return gUtils.directed2undirected(self.edge_list)

    @property
    @cache
    def adjacency_matrix(self) -> npt.NDArray[np.int64]:
        """[N, N], Adjacency matrix"""
        return gUtils.get_weighted_adjacency_matrix(
            self._graph, np.ones(self.num_edges, dtype=np.int64)
        )

    # ----------------------- Couplings, weight of edges ----------------------
    @property
    @cache
    def couplings(self) -> npt.NDArray[np.float64]:
        """[E, ], Coupling constants, in the order of self.edge_list"""
        return self._couplings

    @property
    @cache
    def undirected_couplings(self) -> npt.NDArray[np.float64]:
        """[2E, ], Coupling constants, in the order of self.undirected_edge_list"""
        return gUtils.repeat_weight(self._couplings)

    @property
    @cache
    def weighted_adjacency_matrix(self) -> npt.NDArray[np.float64]:
        """[N, N], Adjacency matrix, weighted by coupling constant"""
        return gUtils.get_weighted_adjacency_matrix(self._graph, self._couplings)

    # ------------------------------- Node types -------------------------------
    @property
    @cache
    def node_types(self) -> list[NodeType]:
        """NodeType of each node"""
        return [node.type for node in self._nodes]

    @property
    @cache
    def is_generator(self) -> npt.NDArray[np.bool_]:
        """True if node is generator"""
        return np.array(
            [node_type is NodeType.GENERATOR for node_type in self.node_types]
        )

    @property
    @cache
    def is_renewable(self) -> npt.NDArray[np.bool_]:
        """True if node is renewable"""
        return np.array(
            [node_type is NodeType.RENEWABLE for node_type in self.node_types]
        )

    @property
    @cache
    def is_consumer(self) -> npt.NDArray[np.bool_]:
        """True if node is consumer"""
        return np.array(
            [node_type is NodeType.CONSUMER for node_type in self.node_types]
        )

    @property
    @cache
    def is_sink(self) -> npt.NDArray[np.bool_]:
        """True if node is sink"""
        return np.array([node_type is NodeType.SINK for node_type in self.node_types])

    @property
    @cache
    def is_source(self) -> npt.NDArray[np.bool_]:
        """True if node is source = generator or renewable"""
        return self.is_generator + self.is_renewable

    @property
    @cache
    def is_user(self) -> npt.NDArray[np.bool_]:
        """True if node is user = consumer or sink"""
        return self.is_consumer + self.is_sink

    # --------------------------------- Nodes -------------------------------------
    @property
    def nodes(self) -> list[Node]:
        """Node instances"""
        return self._nodes

    @property
    def powers(self) -> npt.NDArray[np.float64]:
        """[N, ], Current power of every nodes"""
        return np.array([node.power for node in self._nodes])

    @property
    def masses(self) -> npt.NDArray[np.float64]:
        """[N, ], Current mass of every nodes"""
        return np.array([node.mass for node in self._nodes])

    @property
    def gammas(self) -> npt.NDArray[np.float64]:
        """[N, ], Current gamma of every nodes"""
        return np.array([node.gamma for node in self._nodes])

    @property
    def params(self) -> npt.NDArray[np.float64]:
        """[3, N], stack of (powers, gammas, masses)"""
        return np.stack((self.powers, self.gammas, self.masses))

    @property
    def active_ratios(self) -> npt.NDArray[np.float64]:
        """[N, ], Active ratio of every nodes"""
        return np.array([node.ratio for node in self._nodes])

    @property
    def power_imbalance(self) -> int:
        return sum(node.power for node in self._nodes)

    # ----------------------- Turn on/off ----------------------------
    def turn_off(self) -> None:
        """Turn off the grid: Make all active units to 0"""
        if not self._is_on:
            warnings.warn("Grid is already offline. Do nothing", stacklevel=2)
            return

        self._is_on = False
        for node in self._nodes:
            node.set(0)

    def turn_on(
        self,
        config: TurnOnConfig | None = None,
        num_active_units: list[int] | npt.NDArray[np.int64] | None = None,
    ) -> None:
        """
        Turn on the grid with various strategy.

        Args
        config: See TurnOnConfig documentation
        num_active_units: [N, ], Only used when turn on strategy is manual.
                          Number of active units of each nodes, larger than 1
        """
        if config is None:
            config = GRID_CONFIG.turn_on
        if self._is_on:
            warnings.warn(
                "Grid is already online. Turn off the grid first.", stacklevel=2
            )
            return

        # Turn on the grid
        self._is_on = True
        match config.strategy:
            case "equal":
                turn_on.equal(self._nodes, config.ratio)
            case "random":
                turn_on.random(self._nodes, self._rng)
            case "manual":
                assert num_active_units is not None
                turn_on.manual(self._nodes, num_active_units)
            case "minimum":
                turn_on.minimum(self._nodes)
            case "maximum":
                turn_on.maximum(self._nodes)

        # Power re-balance the grid
        weights = self._rng.choice([-1.0, 1.0], size=self.num_nodes)
        if config.rebalance.strategy != "undirected":
            weights = np.abs(weights)

        balanced = False
        while not balanced:
            balanced = self.rebalance(config.rebalance, weights.copy())

    # ------------------------- Rebalance ------------------------------
    def rebalance(
        self, config: RebalanceConfig, weights: npt.NDArray[np.float32 | np.float64]
    ) -> bool:
        """
        Rebalance grid so that sum of power in entire grid is zero, inplace operation for grid
        If rebalance was failed, return the original state

        Args
        config: See RebalanceConfig documentation
        weights: [N, ] where values indicates different meaning. See rebalance.py

        Return: true if rebalancing was successful
        """
        # No need to do rebalancing
        if self.power_imbalance == 0:
            return True

        # Backup: when rebalance is failed
        active_units = [node.active_units for node in self._nodes]

        _rebalance: rebalance.Rebalancer
        match config.strategy:
            case "undirected":
                _rebalance = rebalance.UndirectedRebalancer()
            case "directed":
                _rebalance = rebalance.DirectedRebalancer()
            case "deterministic":
                _rebalance = rebalance.DeterministicRebalancer()
            case _:
                raise TypeError(f"No such rebalance strategy: {config.strategy}")

        success = _rebalance(self._nodes, weights, config.max_trials, self._rng)
        if not success:
            # Rebalancing is failed. Restore original state
            for node, active_unit in zip(self._nodes, active_units):
                node.set(active_unit)
        return success

    # ----------------------- Power Perturbation -----------------------
    def mark_perturbations(
        self, config: PerturbationConfig | int | None = None
    ) -> npt.NDArray[np.int64]:
        """
        Mark direction and size of perturbation of each consumers/renewables

        Args
        config: perturbation configuration, size of perturbationo

        Return
        perturbation of size [N, ]
            -p : node will be decreased p times
            0 : node will not be perturbated
            +p : node will be increased p times
        """
        if config is None:
            config = GRID_CONFIG.perturbation
        elif isinstance(config, int):
            config = PerturbationConfig(size=config)

        if self.power_imbalance != 0:
            warnings.warn(
                f"Grid is already in power imbalance state. Rebalance it first",
                stacklevel=2,
            )
            return np.array([])
        if not self._is_on:
            warnings.warn("Grid is currently offline.", stacklevel=2)
            return np.array([])

        # Candidates which can be pertubated: consumer, renewables
        candidates = self.is_consumer + self.is_renewable

        # Available perturbation slots for every nodes
        availables = np.array([node.headroom for node in self._nodes], dtype=np.int64)

        # Randomly assign the size of perturbations to candidates
        # Until here, perturbation is all positive representing only it's size, not direction
        distribution = DistributionConfig(name="uniform_wo_avg", delta=config.size)
        while True:
            perturbation = np.zeros(self.num_nodes, dtype=np.int64)
            perturbation[candidates] = sample_restricted_tot(
                distribution,
                config.size,
                candidates.sum(),
                self._rng,
                clip=(0, config.size),
                dtype="int",
            )

            # Check if given perturbation is impossible to nodes
            if np.all(perturbation < availables):
                break

        # Specify direction of the perturbation
        for (idx,) in perturbation.nonzero():
            node, size = self._nodes[idx], perturbation[idx]

            # Check edge case: when node should be perturbated in certain direction
            if size > node.headroom_increase:
                # Cannot increase
                perturbation[idx] *= -1
                continue
            elif size > node.headroom_decrease:
                # Cannot decrease
                continue

            # Randomly select direction
            if self._rng.random() < 0.5:
                perturbation[idx] *= -1

        return perturbation

    def perturbate(self, perturbations: npt.NDArray[np.int64]) -> None:
        """
        Increase/Decrease active units of nodes according to perturbation \\
        Othere states such as phase, dphase of nodes are not changing

        Args
        perturbations: return of Grid.mark_perturbation
        """
        for node, perturbation in zip(self.nodes, perturbations):
            if perturbation == 0:
                continue
            elif perturbation < 0:
                [node.decrease() for _ in range(abs(perturbation))]
            else:
                [node.increase() for _ in range(abs(perturbation))]

    # ----------------------- Swing dynamics -----------------------
    def run(
        self,
        config: SwingConfig,
        phase: npt.NDArray[np.float64] | None = None,
        dphase: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Run swing equation at the given grid

        Args
        grid: target grid to find steady state \\
        config: Configuration for swing solving. Seed Swingconfig documentation \\
        phase: [N, ]. initial phase. If not given, zero \\
        dphases: [N, ]. initial dphase. If not given, zero \\
        full_trajectory: If true, return full trajectory of finding steady state

        Return: [S+1, N], [S+1, N] where S is number of time steps. \\
                Note that initial condition is included
        """

        def normalize_phase(phase: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """Normalize phase into [-pi, pi)"""
            return (phase + np.pi) % (2 * np.pi) - np.pi

        if not self._is_on:
            warnings.warn("Grid is currently offline.", stacklevel=2)
            return np.array([]), np.array([])

        # Setup initial condition
        if phase is None:
            phase = np.zeros(self.num_nodes)
        if dphase is None:
            dphase = np.zeros(self.num_nodes)
        assert len(phase) == self.num_nodes
        assert len(dphase) == self.num_nodes

        # Create swing solver and prepare initial condition
        swing_solver = get_swing_solver(
            config, self.weighted_adjacency_matrix, self.params
        )

        # Create monitor to check if the swing solving should be early stop or not
        monitor = get_monitor(config.monitor)

        # Container to store full trajectory
        time = 0.0
        phases, dphases = [phase.copy()], [dphase.copy()]

        # Solve swing equation
        while time < config.max_time:
            time += config.dt
            phase, dphase = swing_solver(phase=phase, dphase=dphase)

            phases.append(phase.copy())
            dphases.append(dphase.copy())

            if monitor(phase, dphase, config.dt):
                break

        # Return
        return normalize_phase(np.stack(phases)), np.stack(dphases)
