"""Zero-Model Intelligence (zeromodel) - Standalone package for cognitive compression"""

from .core import ZeroModel
from .sorter import TaskSorter
from .normalizer import DynamicNormalizer
from .transform import transform_vpm, get_critical_tile
from .edge import EdgeProtocol

__version__ = "1.0.0"
__all__ = [
    'ZeroModel',
    'TaskSorter',
    'DynamicNormalizer',
    'transform_vpm',
    'get_critical_tile',
    'EdgeProtocol'
]