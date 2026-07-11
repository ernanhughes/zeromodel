# Spatial calculus

ZeroModel's dense-representation thesis has two parts:

1. a source table can hold many signals at once;
2. a view can reorganize that dense table so the relevant signal becomes spatially salient.

`ViewProfile` implements the manual form of this idea. `SpatialOptimizer` implements the first learned form: it searches for metric weights that concentrate scored mass into a top-left inspection region.

## From manual view to optimized view

Manual view:

```text
human chooses metric weights
→ ViewProfile
→ VPM artifact
```

Optimized view:

```text
source table or table series
→ explicit top-left mass objective
→ learned metric weights
→ ViewProfile
→ VPM artifact
```

The output is still a normal ZeroModel view. That matters because source mapping, provenance, rendering, bundling, and region gates continue to work.

## What this PR validates

This module validates a narrow claim:

> ZeroModel can derive a deterministic metric-weight view profile that improves an explicit top-left mass objective over scored tables.

It does not validate:

- automatic task understanding;
- universal best-view discovery;
- decision accuracy improvement;
- planet-scale traversal;
- learned 40-step routing over world data.

Those are research directions that require benchmark fixtures and raw results.

## Relationship to the older tools

The older `spatial_optimizer.py` sketch contained the right ingredients: top-left mass concentration, dual ordering, metric graphs, canonical layouts, and temporal series. The cleaned core version narrows that into a package-safe primitive:

```text
SpatialOptimizer.fit(...)
→ SpatialOptimizationResult
→ ViewProfile
```

The older `decision_manifold.py` sketch points to the next stage: sequences of optimized views, curvature, inflection points, and decision rivers. That should remain separate until there are tests and a clear artifact model for temporal manifolds.

## Research direction

The immediate benchmark question is:

> Given the same dense source table, do optimized views concentrate relevant signal better than manual or source-order views?

A later, stronger question is:

> Does increased top-left concentration improve human or automated inspection outcomes?

The first is geometric and can be tested now. The second is behavioral and requires a task benchmark.
