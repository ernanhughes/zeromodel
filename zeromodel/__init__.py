# zeromodel/__init__.py
"""
Zero-Model Intelligence (zeromodel) - Standalone package for cognitive compression
"""

from .config import init_config
from .core import ZeroModel
from .edge import EdgeProtocol
from .hierarchical import HierarchicalVPM
from .hierarchical_edge import HierarchicalEdgeProtocol
from .normalizer import DynamicNormalizer
from .transform import get_critical_tile, transform_vpm
from .ppmi import (    PPMImageWriter,
    PPMImageReader,
    build_parent_level_png,
    AGG_MAX,
)

__version__ = "1.0.4"
__all__ = [
    "ZeroModel",
    "init_config",
    "HierarchicalVPM",
    "DynamicNormalizer",
    "transform_vpm",
    "get_critical_tile",
    "EdgeProtocol",
    "HierarchicalEdgeProtocol",
    "PPMImageWriter",
    "PPMImageReader",
    "build_parent_level_png",
    "AGG_MAX",
]

