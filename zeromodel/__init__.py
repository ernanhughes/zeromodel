# zeromodel/__init__.py
"""
Zero-Model Intelligence (zeromodel) - Standalone package for cognitive compression
"""

from .core import ZeroModel
from .config import init_config
from .hierarchical import HierarchicalVPM
from .edge import EdgeProtocol
from .hierarchical_edge import HierarchicalEdgeProtocol
from .normalizer import DynamicNormalizer
from .transform import get_critical_tile, transform_vpm

__version__ = "1.0.4"
__all__ = [
    "ZeroModel",
    "init_config",
    "HierarchicalVPM",
    "DynamicNormalizer",
    "transform_vpm",
    "get_critical_tile",
    "EdgeProtocol",
    "HierarchicalEdgeProtocol"
]

