#  zeromodel/__init__.py
"""
Zero-Model Intelligence (zeromodel) - Standalone package for cognitive compression
"""
from __future__ import annotations

from .config import init_config
from .core import ZeroModel
from .edge import EdgeProtocol
from .hierarchical import HierarchicalVPM
from .hierarchical_edge import HierarchicalEdgeProtocol
from .normalizer import DynamicNormalizer
from .vpm.image import (AGG_MAX, VPMImageReader, VPMImageWriter,
                        build_parent_level_png)
from .vpm.transform import get_critical_tile, transform_vpm

__version__ = "1.0.6"
__all__ = [
    "ZeroModel",
    "init_config",
    "HierarchicalVPM",
    "DynamicNormalizer",
    "transform_vpm",
    "get_critical_tile",
    "EdgeProtocol",
    "HierarchicalEdgeProtocol",
    "VPMImageWriter",
    "VPMImageReader",
    "build_parent_level_png",
    "AGG_MAX",
]

