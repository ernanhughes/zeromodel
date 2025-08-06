"""Zero-Model Intelligence (zeromodel) - Standalone package for cognitive compression"""

from .core import HierarchicalVPM, ZeroModel
from .edge import EdgeProtocol
from .normalizer import DynamicNormalizer
from .transform import get_critical_tile, transform_vpm

__version__ = "1.0.4"
__all__ = [
    "ZeroModel",
    "HierarchicalVPM",
    "DynamicNormalizer",
    "transform_vpm",
    "get_critical_tile",
    "EdgeProtocol",
]

