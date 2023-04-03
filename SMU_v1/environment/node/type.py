from enum import Enum, auto


class NodeType(Enum):
    GENERATOR = auto()
    RENEWABLE = auto()
    CONSUMER = auto()
