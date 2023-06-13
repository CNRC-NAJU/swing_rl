import inspect

from config.grid import ResetConfig

from .create import (create_couplings, create_graph, create_node_types,
                     create_nodes)
from .grid import Grid


def reset(
    grid: Grid, config: ResetConfig | dict[str, bool], **kwargs
) -> Grid:
    """
    Create new grid, while preserving some of previous grid information

    Args
    grid: Grid to be resetted
    config: True/False for each graph, couplings, node_types, nodes
    kwargs: keyword arguments for create functions

    Return
    grid: new node. Note that returnning grid is offline
    """
    if isinstance(config, dict):
        config = ResetConfig(**config)

    # When couplings and nodes are not reset, then there is no need to do reset
    if not (config.coupling or config.node):
        grid.turn_off()
        return grid

    # Create new graph/couplings/node_types/nodes or extract it from previous grid
    if config.graph:
        valid_kwargs = inspect.getfullargspec(create_graph).kwonlyargs
        graph_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}
        graph = create_graph(grid.rng, **graph_kwargs)
    else:
        graph = grid.graph

    if config.coupling:
        valid_kwargs = inspect.getfullargspec(create_couplings).kwonlyargs
        couplings_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}
        couplings = create_couplings(
            graph.number_of_edges(), grid.rng, **couplings_kwargs
        )
    else:
        couplings = grid.couplings

    if config.node_type:
        valid_kwargs = inspect.getfullargspec(create_node_types).kwonlyargs
        node_types_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}
        node_types = create_node_types(
            graph.number_of_nodes(), grid.rng, **node_types_kwargs
        )
    else:
        node_types = grid.node_types

    if config.node:
        valid_kwargs = inspect.getfullargspec(create_nodes).kwonlyargs
        nodes_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}
        nodes = create_nodes(node_types, grid.rng, **nodes_kwargs)
    else:
        nodes = grid.nodes

    # Create new grid
    return Grid(grid.rng, graph, couplings, node_types, nodes)
