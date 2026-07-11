# Temporal spatial calculus

ZeroModel now has three connected layers:

```text
Dense source table
→ ViewProfile
→ SpatialOptimizer
→ DecisionManifold
```

The decision manifold is the temporal layer. It treats a run, trace, training history, evaluator stream, or other scored sequence as a series of dense panels.

## Concept

A source table can contain many signals at once. A view profile turns one policy lens up. A spatial optimizer derives a metric-weight profile for an explicit top-left mass objective.

A decision manifold asks what happens when those panels evolve over time.

```text
panel_0 → optimized_view_0
panel_1 → optimized_view_1
panel_2 → optimized_view_2
...
```

Each optimized panel is still a normal VPM artifact. The manifold records how the spatial view changes between adjacent frames.

## What is measured

The first implementation measures:

| Signal | Meaning |
|---|---|
| `top_left_mass` | how much normalized signal sits in the inspection region |
| `metric_weights` | which metric lens the optimizer derived for that frame |
| `row_order` | which source rows became most salient |
| `column_order` | which metric columns became most salient |
| `curvature` | weighted change across mass, weights, row order, and column order |
| `inflection_indices` | frames after the largest adjacent-frame changes |

This is intentionally geometric, not semantic. It does not say *why* the change happened. It surfaces *where* inspection should begin.

## Relation to the old “40 steps” idea

The older idea was that a very large sequence of panels could be reduced to a small number of meaningful inspection steps. This module is the first reproducible version of that idea:

```text
many dense panels
→ optimized spatial views
→ curvature over time
→ candidate inspection frames
```

The repository still should not claim that it can find the universal best forty steps for arbitrary world data. The safe claim is narrower:

> ZeroModel can summarize a sequence of dense scored panels as a deterministic decision manifold and surface frames with large spatial-view changes.

## What remains to prove

Before making stronger claims, we need:

1. generated temporal fixtures with known change points;
2. benchmark comparisons against scalar dashboards or random frame sampling;
3. real exported sequences from training, critics, RAG verification, or Writer traces;
4. user studies or task metrics showing inspection improvement.

## Why this matters

This completes the first foundation:

```text
static dense representation
→ multiple policy views
→ optimized view profiles
→ temporal decision manifolds
```

Applications such as hallucination inspection, training progress, critic traces, and Writer evaluator streams can now sit on top of that foundation.
