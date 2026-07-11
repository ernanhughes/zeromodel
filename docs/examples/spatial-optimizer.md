# Spatial optimizer

The spatial optimizer is the first core bridge from manual dense views to spatial calculus.

A `ViewProfile` says which metrics to turn up or down. A `SpatialOptimizer` learns a non-negative metric-weight profile for one explicit objective:

> concentrate high-signal mass in the top-left inspection region.

It does **not** prove that the resulting view is semantically correct, universally optimal, or better for every task. It optimizes a deterministic geometric objective over scored data.

## Example

```python
from zeromodel import ScoreTable, SpatialOptimizer, build_optimized_view, optimize_view_profile

source = ScoreTable(
    values=[
        [0.10, 0.50, 0.20],
        [0.95, 0.50, 0.25],
        [0.90, 0.50, 0.15],
        [0.05, 0.50, 0.20],
    ],
    row_ids=["background", "target_a", "target_b", "flat"],
    metric_ids=["target", "constant", "weak"],
)

optimizer = SpatialOptimizer(Kc=2, Kr=2, alpha=0.95, max_iters=40)
result = optimize_view_profile(source, name="optimized-target", optimizer=optimizer)
view = build_optimized_view(source, name="optimized-target", optimizer=optimizer)

print(result.metric_weights)
print(result.baseline_mass, result.optimized_mass)
print(view.cell(0, 0).row_id, view.cell(0, 0).metric_id)
```

## What is optimized

For a candidate metric-weight vector, the optimizer:

1. normalizes each metric to `[0, 1]`,
2. sorts columns by learned weight,
3. sorts rows by weighted intensity in the top `Kc` columns,
4. computes spatially decayed mass in the top `Kr × Kc` block,
5. uses deterministic coordinate ascent to improve that score.

## Why it matters

This turns the old spatial-calculus idea into a repo-backed primitive:

```text
same dense source table
→ learned metric weights
→ ViewProfile
→ optimized VPM view
→ same source digest and source mapping
```

The next step after this is not to claim universal optimization. The next step is benchmark fixtures that compare manual views, optimized views, and task-specific outcomes.
