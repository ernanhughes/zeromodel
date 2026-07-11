# Decision manifold

A single VPM is a deterministic spatial view over one dense source table. A decision manifold is a sequence of such panels over time.

This module answers a narrow geometric question:

> Given a sequence of dense scored panels, where did the optimized spatial view change most?

It does not claim semantic understanding, automatic causal explanation, or improved decision accuracy. It tracks deterministic changes in an already-scored representation.

## Use

```python
from zeromodel import ScoreTable, SpatialOptimizer, build_decision_manifold

panels = [
    ScoreTable(
        values=[
            [0.20, 0.60, 0.10],
            [1.00, 0.10, 0.10],
            [0.10, 0.20, 0.30],
        ],
        row_ids=["forest", "crowd", "traffic"],
        metric_ids=["people", "trees", "risk"],
    ),
    ScoreTable(
        values=[
            [0.15, 0.55, 0.14],
            [0.25, 0.10, 0.30],
            [0.10, 0.18, 1.00],
        ],
        row_ids=["forest", "crowd", "traffic"],
        metric_ids=["people", "trees", "risk"],
    ),
]

summary = build_decision_manifold(
    panels,
    optimizer=SpatialOptimizer(Kc=1, Kr=1),
    name="scene-shift",
    inflection_top_k=1,
)

print(summary.inflection_indices)
print(summary.mass_series)
print(summary.curvature_series)
```

## What gets built

For each panel, the manifold:

1. runs `SpatialOptimizer.fit()`;
2. emits a normal `ViewProfile`;
3. builds a normal `VPMArtifact`;
4. records top-left mass, metric weights, row order, and column order.

Then it compares adjacent frames using:

- top-left mass change;
- metric-weight movement;
- row-order movement;
- column-order movement.

The largest changes become `inflection_indices`.

## Example script

```bash
python examples/research_decision_manifold.py
```

This writes per-frame PNG/SVG views and a summary JSON under `.zeromodel-demo/`.

## Safe wording

Valid:

> ZeroModel can summarize a sequence of dense scored panels as a deterministic decision manifold and surface frames with large spatial-view changes.

Invalid:

> ZeroModel understands why the world changed.

Invalid:

> ZeroModel finds the forty objectively important moments in any dataset.

Those require task-specific benchmarks and human/user validation.
