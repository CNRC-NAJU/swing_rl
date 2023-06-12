from config.grid import ResetConfig

from .grid import Grid


def reset(
    grid: Grid, config: ResetConfig | dict[str, bool]
) -> Grid:
    """
    Create new grid, while preserving some of previous grid information

    Args
    grid: Grid to be resetted
    config: True/False for each graph, couplings, node_types, nodes

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
    graph = None if config.graph else grid.graph
    couplings = None if config.coupling else grid.couplings
    node_types = None if config.node_type else grid.node_types
    nodes = None if config.node else grid.nodes

    # Create new grid
    return Grid(grid.rng, graph, couplings, node_types, nodes)
