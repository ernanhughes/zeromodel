# View profiles

A ZeroModel source table is dense: it can contain many signals for the same set of rows.

A `ViewProfile` is a policy lens over that table. It turns selected metrics up or down and builds a deterministic VPM view while preserving the same underlying source evidence.

```python
from zeromodel import ScoreTable, ViewProfile, build_view

source = ScoreTable(
    values=[
        [0.10, 0.96, 0.05, 0.72, 0.20],
        [0.94, 0.12, 0.08, 0.18, 0.35],
        [0.24, 0.07, 0.97, 0.08, 0.78],
        [0.07, 0.18, 0.04, 0.98, 0.10],
    ],
    row_ids=["forest", "crowd", "traffic", "meadow"],
    metric_ids=["people", "trees", "cars", "grass", "risk"],
)

people_view = build_view(source, ViewProfile.from_metric("people", name="people"))
tree_view = build_view(source, ViewProfile.from_metric("trees", name="trees"))
risk_view = build_view(source, ViewProfile.from_metric("risk", name="risk"))

assert people_view.source.digest == tree_view.source.digest == risk_view.source.digest
assert people_view.cell(0, 0).row_id == "crowd"
assert tree_view.cell(0, 0).row_id == "forest"
assert risk_view.cell(0, 0).row_id == "traffic"
```

The source data did not change. The view changed.

## Turning metrics up and down

Positive metric weights make high values salient. Negative weights make low values salient.

```python
from zeromodel import ViewProfile, build_view

safe_open_space = ViewProfile(
    name="safe-open-space",
    metric_weights={
        "grass": 1.0,
        "risk": -0.8,
        "cars": -0.4,
    },
)

view = build_view(source, safe_open_space)
```

This profile prefers high `grass`, low `risk`, and low `cars`.

## Why this is core

View profiles make the dense-representation claim concrete:

```text
same source table
+ different view profile
= different deterministic VPM view
+ same source digest
+ same source/cell mapping
```

This is not a model re-run and not a new evidence extraction step. It is a deterministic reorganization of the existing dense artifact.

## Related example

```bash
python examples/research_multiview_dense_artifact.py
```

The example writes bundles, PNGs, SVGs, and a summary under `.zeromodel-demo/multiview_dense_artifact/`.
