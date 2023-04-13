from enum import Enum


class NodeType(Enum):
    GENERATOR = 0
    RENEWABLE = 1
    CONSUMER = 2
    CONTROLLABLE_CONSUMER = 3
