"""Zero-Model Intelligence (zeromodel) - Standalone package for cognitive compression"""

from .config import get_config_value, load_config, validate_config
from .core import HierarchicalVPM, ZeroModel
from .normalizer import DynamicNormalizer
from .transform import get_critical_tile, transform_vpm
from .edge import EdgeProtocol

__version__ = "1.0.4"
__all__ = [
    "ZeroModel",
    "HierarchicalVPM",
    "DynamicNormalizer",
    "transform_vpm",
    "get_critical_tile",
    "load_config",
    "get_config_value",
    "validate_config",
    "EdgeProtocol",
]

