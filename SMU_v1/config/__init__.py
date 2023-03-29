from .consumer import ConsumerConfig
from .distribution import DistributionConfig
from .generator import GeneratorConfig
from .graph import GraphConfig
from .grid import GridConfig
from .renewable import RenewableConfig
from .rk import RKConfig
from .rl import RLConfig


def validate_config():
    """Create each configuration once to validate their values"""
    ConsumerConfig()
    DistributionConfig()
    GeneratorConfig()
    GraphConfig()
    GridConfig()
    RenewableConfig()
    RKConfig()
    RLConfig()


validate_config()
