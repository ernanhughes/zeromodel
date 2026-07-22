# zeromodel-navigation

A finite, deterministic hierarchy compiler and traversal engine over
identified artifacts.

## This is not Search

This package compiles and traverses a **finite, closed** hierarchy of
already-identified artifacts. It does not define similarity, relevance, or
nearest-neighbour retrieval. Traversal rules are stable, declared
descriptors with production implementations that route on explicit,
declared criteria (a request attribute, a declared priority, a fixed
target range) - never a learned or heuristic "closest match."

A later Search package may implement similarity-driven `TraversalRule`s
against the same `TraversalRule` protocol this package defines; that is out
of scope here.

## Two separate hierarchy concepts

- The existing **intra-artifact pyramid** (`zeromodel.analysis.hierarchy`,
  `build_pyramid`) reduces a single VPM field into coarser levels of
  itself. Untouched by this package.
- The **artifact corpus hierarchy** this package compiles is a tree over
  *many* identified artifacts: a root tile, internal navigation tiles, and
  leaf bindings to individual artifacts.

## Dependency rule

`navigation -> core + artifacts` only. Navigation does not depend on, or
import, Trust. Secure applications may compose Navigation and Trust
without coupling the packages directly - see the integration seam example
below.

## Persistence

Navigation defines no store or repository of its own. Every tile and leaf
binding is persisted as canonical bytes through the Artifacts package's
`ArtifactStore`/`ArtifactResolver` protocol; Navigation owns the DTUs'
semantics, Artifacts owns storage and identity.

## Public API

`HierarchyManifestDTO`, `NavigationTileDTO`, `HierarchyCompilerSpecDTO`,
`TraversalRule`, `TraversalRequestDTO`, `TraversalStepDTO`,
`TraversalResultDTO`, `TraversalReceiptDTO`, `compile_hierarchy`,
`validate_hierarchy`, `traverse`, `replay_traversal`.

Supporting DTOs (`TileCoverageDTO`, `TilePointerDTO`, `LeafBindingDTO`,
`TraversalRuleDescriptorDTO`, `TraversalFailureDTO`) are available from
`zeromodel.navigation.dto` for tests and advanced composition, but are not
part of the curated top-level surface.

## Claims boundary

The supported claim is: compiling and deterministically traversing a
finite, identified hierarchy with complete artifact resolution and a
replayable trace.

This package does **not** claim: planet-scale hierarchies, infinite
in-memory capacity, logarithmic-time guarantees, semantic search,
nearest-neighbour retrieval, "40-hop world navigation," or
storage-independent performance.

## Integration seam: composing with Trust (without depending on it)

Navigation never imports `zeromodel.trust`. A secure application composes
both packages at the call site:

```python
from zeromodel.navigation import compile_hierarchy, validate_hierarchy, traverse
from zeromodel.trust import verify_artifact_for_scope, require_authorized

# 1. Resolve the hierarchy root (Navigation + Artifacts only).
root_manifest = ...  # HierarchyManifestDTO, previously compiled

# 2. Verify the root/hierarchy is authorized before trusting its contents
#    (Trust only - Navigation is not involved in this step).
decision = verify_artifact_for_scope(
    artifact_ref=root_manifest.root_ref,
    ...,
)
require_authorized(decision)  # raises ArtifactNotAuthorized if not

# 3. Validate structural closure (Navigation only).
validate_hierarchy(root_manifest, store=artifact_store)

# 4. Traverse (Navigation only).
result = traverse(manifest=root_manifest, store=artifact_store, rule=rule, request=request)

# 5. Optionally, verify each resolved leaf artifact through Trust before use.
```

This example is illustrative only - it does not execute a real controller
and makes no safety-certification claim. A safety-critical caller must not
skip step 2 or step 5.
