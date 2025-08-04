"""Zero-Model Intelligence (zeromodel) - Standalone package for cognitive compression"""

from .config import get_config_value, load_config, validate_config
from .core import HierarchicalVPM, ZeroModel
from .normalizer import DynamicNormalizer
from .sorter import TaskSorter
from .transform import get_critical_tile, transform_vpm

__version__ = "1.0.0"
__all__ = [
    'ZeroModel',
    'HierarchicalVPM',
    'TaskSorter',
    'DynamicNormalizer',
    'transform_vpm',
    'get_critical_tile',
    'load_config',
    'get_config_value',
    'validate_config'
]

__version__ = "1.0.0"
__all__ = [
    'ZeroModel',
    'HierarchicalVPM',
    'TaskSorter',
    'DynamicNormalizer',
    'transform_vpm',
    'get_critical_tile',
    'EdgeProtocol',
]