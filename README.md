# ZeroModel

**ZeroModel turns scored data into deterministic, inspectable Visual Policy Map artifacts and small consumers that can operate without a model at decision time.**

A VPM is a deterministic spatial view over a table of scored items. It carries values, stable row and metric identifiers, a layout recipe, view ordering, source mapping, provenance, and deterministic identity.

The 1.0.13 release candidate is split into six namespace-package distributions.
Import from the package namespaces directly; the legacy root compatibility surface
is intentionally absent.

Public claims are tracked in [`docs/claims-audit.md`](docs/claims-audit.md). Treat that file as the source of truth for what is validated, what is implemented with thin evidence, and what remains a roadmap claim.

## Install

Current GitHub install:

```bash
python -m pip install "git+https://github.com/ernanhughes/zeromodel.git@main"
```

For the 1.0.13 release candidate, install the pieces you need:

```bash
python -m pip install \
  zeromodel==1.0.13 \
  zeromodel-analysis==1.0.13 \
  zeromodel-observation==1.0.13 \
  zeromodel-vision==1.0.13 \
  zeromodel-video==1.0.13 \
  zeromodel-sqlalchemy==1.0.13
```

For development:

```bash
python -m pip install -e packages/core -e packages/analysis -e packages/observation -e packages/vision -e packages/video -e packages/sqlalchemy
python scripts/run_fast_tests.py
python scripts/validate_release_candidate.py
```

Release steps are documented in [`docs/release.md`](docs/release.md).

## Core artifact

```python
from zeromodel.core import LayoutRecipe, ScoreTable, build_vpm

score_table = ScoreTable(
    values=[[0.9, 0.2], [0.4, 0.8]],
    row_ids=["candidate-a", "candidate-b"],
    metric_ids=["quality", "uncertainty"],
)

recipe = LayoutRecipe.from_dict({
    "version": "vpm-layout/0",
    "name": "quality-first",
    "row_order": {
        "kind": "lexicographic",
        "keys": [{"metric_id": "quality", "direction": "desc"}],
        "tie_break": "row_id",
    },
    "column_order": {"kind": "source"},
    "normalization": {"kind": "per_metric_minmax", "clip": True},
})

artifact = build_vpm(score_table, recipe)
cell = artifact.cell(view_row=0, view_column=0)
region = artifact.region(rows=slice(0, 1), columns=slice(0, 2))
```

## Capability surface

| Capability | Module |
|---|---|
| Immutable artifact kernel | `zeromodel.core` |
| Analysis, views, spatial/manifold, policy diagnostics | `zeromodel.analysis` |
| Observation DTOs and visual address contracts | `zeromodel.observation` |
| Deterministic visual index and visual policy reader | `zeromodel.vision` |
| Video policy, arcade fixture, video action-set DTOs/stores | `zeromodel.video` |
| Explicit SQLAlchemy persistence runtime | `zeromodel.persistence.sqlalchemy` |

## Policy lookup: signs, not directions

`VPMPolicyLookup` is the small 1.0 consumer behind the blog phrase “signs, not directions.” Rows are discretized runtime states, metric columns are candidate actions, and the consumer returns the winning action plus the exact VPM cell that produced it.

```python
from zeromodel.core import LayoutRecipe, ScoreTable, VPMPolicyLookup, build_vpm

source = ScoreTable(
    values=[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    row_ids=["state:left", "state:right", "state:aligned"],
    metric_ids=["LEFT", "RIGHT", "STAY", "FIRE"],
)
recipe = LayoutRecipe.from_dict({
    "version": "vpm-layout/0",
    "name": "policy-source-order",
    "row_order": {"kind": "source", "tie_break": "row_id"},
    "column_order": {"kind": "source"},
    "normalization": {"kind": "per_metric_minmax", "clip": True},
})

artifact = build_vpm(source, recipe)
decision = VPMPolicyLookup(artifact).read("state:aligned")

assert decision.action == "FIRE"
assert decision.artifact_id == artifact.artifact_id
```

The demo is a tiny arcade shooter:

```bash
python examples/arcade_shooter_policy.py
```

It compiles a closed-world shooter policy into one VPM artifact, then replays by reading state signs from that artifact. Tests assert wave clear, random-baseline comparison, and deterministic action trace replay.

See [`docs/examples/sign-reader.md`](docs/examples/sign-reader.md).

## Criticality-aware verification

ZeroModel 1.0.11 can add two separate evidence metrics to a Q-bearing policy surface:

```text
criticality     = best action value - worst action value
decision margin = best action value - second-best action value
```

Criticality estimates how consequential a poor choice could be. Decision margin measures how decisively the winner beats its nearest alternative. Describe the first metric as VIPER-style criticality only when the source values are Q-values or an equivalent consequence-bearing teacher signal.

```python
from zeromodel.analysis import (
    PolicyPropertyChecker,
    PolicyPropertySpec,
    with_q_diagnostics,
)
from zeromodel.core import VPMPolicyLookup, build_vpm

ACTIONS = ("LEFT", "RIGHT", "STAY", "FIRE")

enriched = with_q_diagnostics(
    source,
    action_metric_ids=ACTIONS,
)
artifact = build_vpm(enriched, recipe)

# Diagnostic metadata lets the reader safely separate actions from evidence.
reader = VPMPolicyLookup(artifact)
```

Evidence metrics are returned with the decision but never participate in action selection.

A finite policy property is declarative and versioned:

```python
fire_requires_alignment = PolicyPropertySpec.from_dict({
    "id": "fire_requires_alignment",
    "version": "1",
    "assert": {
        "implies": [
            {"eq": [{"var": "winner"}, "FIRE"]},
            {"all": [
                {"eq": [{"var": "state.tank"}, {"var": "state.target"}]},
                {"eq": [{"var": "state.cooldown"}, 0]},
            ]},
        ]
    },
})

report = PolicyPropertyChecker(
    artifact,
    action_metric_ids=ACTIONS,
    evidence_metric_ids=("criticality", "decision_margin"),
).check([fire_requires_alignment])

verification_artifact = report.to_vpm()
```

The verification artifact points back to the exact checked policy through a provenance parent with relation `verifies`. Failed checks retain exact counterexample rows, candidates, evidence, and source/view coordinates.

Run the full counterexample, repair, and re-verification fixture:

```bash
python examples/criticality_verification.py \
  --output-dir docs/assets/criticality-verification
```

See [`docs/examples/criticality-verification.md`](docs/examples/criticality-verification.md) and [`docs/research/viper-policy-compilation.md`](docs/research/viper-policy-compilation.md).

## Dense view profiles

A source table can contain many signals at once. A view profile is a policy lens over that dense table: turn up one set of metrics and the matching rows/columns become salient without changing the source evidence.

```python
from zeromodel.core import ScoreTable, ViewProfile, build_view

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
print(people_view.cell(0, 0).row_id)  # crowd
print(tree_view.cell(0, 0).row_id)    # forest
print(risk_view.cell(0, 0).row_id)    # traffic
```

Positive weights make high values salient. Negative weights make low values salient.

See [`docs/examples/view-profiles.md`](docs/examples/view-profiles.md) and [`docs/research/dense-multiview-representation.md`](docs/research/dense-multiview-representation.md).

## Spatial optimizer

The spatial optimizer derives a `ViewProfile` for one explicit geometric objective: concentrate high-signal mass in the top-left inspection region.

```python
from zeromodel.analysis import SpatialOptimizer, build_optimized_view, optimize_view_profile
from zeromodel.core import ScoreTable

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

optimizer = SpatialOptimizer(Kc=2, Kr=2, alpha=0.95, max_evals=40)
result = optimize_view_profile(source, name="optimized-target", optimizer=optimizer)
view = build_optimized_view(source, name="optimized-target", optimizer=optimizer)

print(result.baseline_mass, result.optimized_mass)
print(view.cell(0, 0).row_id, view.cell(0, 0).metric_id)
```

This does not claim the optimizer learns the correct semantic view for every task. It proves a deterministic top-left mass objective can emit a normal `ViewProfile` while preserving source mapping.

See [`docs/examples/spatial-optimizer.md`](docs/examples/spatial-optimizer.md) and [`docs/research/spatial-calculus.md`](docs/research/spatial-calculus.md).

## Decision manifold

A decision manifold turns a sequence of dense scored panels into optimized VPM frames, then surfaces where the spatial view changes most.

```python
from zeromodel.analysis import SpatialOptimizer, build_decision_manifold
from zeromodel.core import ScoreTable

panels = [
    ScoreTable(
        values=[[0.20, 0.60, 0.10], [1.00, 0.10, 0.10], [0.10, 0.20, 0.30]],
        row_ids=["forest", "crowd", "traffic"],
        metric_ids=["people", "trees", "risk"],
    ),
    ScoreTable(
        values=[[0.15, 0.55, 0.14], [0.25, 0.10, 0.30], [0.10, 0.18, 1.00]],
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

This does not claim semantic cause or universal change-point discovery. It provides deterministic temporal geometry over scored panels.

See [`docs/examples/decision-manifold.md`](docs/examples/decision-manifold.md) and [`docs/research/temporal-spatial-calculus.md`](docs/research/temporal-spatial-calculus.md).

## PHOS and edge usage

```python
from zeromodel.analysis import TopLeftGate, guarded_pack_artifact
from zeromodel.core import write_png

packed = guarded_pack_artifact(artifact)
write_png(packed.packed, "artifact_phos.png")
result = TopLeftGate(threshold=0.75).evaluate(packed.packed)
print(result.accepted, result.score)
```

## Learning trace usage

Tracking means a score moved. Learning means a feedback-driven change improves corrected work, transfers to held-out work, and avoids unacceptable regression.

```python
from zeromodel.analysis import LearningObservation, build_learning_vpm

assessment = build_learning_vpm([
    LearningObservation("claim-support", before=0.42, after=0.72, split="train"),
    LearningObservation("related-claim", before=0.50, after=0.63, split="heldout"),
    LearningObservation("summary-quality", before=0.82, after=0.81, split="regression"),
])

print(assessment.learned)
learning_artifact = assessment.artifact
```
