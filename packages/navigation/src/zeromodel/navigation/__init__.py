"""zeromodel-navigation: finite, deterministic artifact-corpus hierarchy.

Depends only on `zeromodel` (core) and `zeromodel-artifacts`. Does not
depend on, or import, `zeromodel.trust`.

The public surface is deliberately restricted to the 12 names in
`__all__`. Supporting DTOs (`TileCoverageDTO`, `TilePointerDTO`,
`LeafBindingDTO`, `TraversalRuleDescriptorDTO`, `TraversalFailureDTO`) and
the reference rule implementations remain available from
`zeromodel.navigation.dto` / `zeromodel.navigation.rules` for tests and
advanced composition.
"""

from zeromodel.navigation.compiler import compile_hierarchy, validate_hierarchy
from zeromodel.navigation.dto import (
    HierarchyCompilerSpecDTO,
    HierarchyManifestDTO,
    NavigationTileDTO,
    TraversalReceiptDTO,
    TraversalRequestDTO,
    TraversalResultDTO,
    TraversalStepDTO,
)
from zeromodel.navigation.rules import TraversalRule
from zeromodel.navigation.traversal import replay_traversal, traverse

__all__ = [
    "HierarchyCompilerSpecDTO",
    "HierarchyManifestDTO",
    "NavigationTileDTO",
    "TraversalRequestDTO",
    "TraversalResultDTO",
    "TraversalReceiptDTO",
    "TraversalRule",
    "TraversalStepDTO",
    "compile_hierarchy",
    "replay_traversal",
    "traverse",
    "validate_hierarchy",
]
