"""ZeroModel v2 artifact kernel.

This package contains the first-principles reference implementation for the
Visual Policy Map artifact contract. It is intentionally separate from the v1
experimental package surface.
"""
from __future__ import annotations

from .artifact import (
    LayoutRecipe,
    ScoreTable,
    VPMArtifact,
    VPMCell,
    VPMRegion,
    build_vpm,
)

__all__ = [
    "LayoutRecipe",
    "ScoreTable",
    "VPMArtifact",
    "VPMCell",
    "VPMRegion",
    "build_vpm",
]
