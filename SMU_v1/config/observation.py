from dataclasses import dataclass


@dataclass
class ObservationConfig:
    node_type: bool = True
    phase: bool = True
    dphase: bool = True
    mass: bool = True
    gamma: bool = True
    power: bool = True
    active_ratio: bool = True
    perturbation: bool = True
    edge_list: bool = True
    coupling: bool = True
