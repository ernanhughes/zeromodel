# ZeroModel Core

`zeromodel` is the lightweight core distribution for ZeroModel 1.0.13. It owns the `zeromodel.core` import namespace.

Core includes deterministic VPM artifact construction, stable artifact and matrix identities, basic views, `.vpm` bundle serialization, lightweight PNG/SVG rendering, exact bounded policy lookup, and Lua export for compiled policy plans.

Core deliberately excludes analysis, observation contracts, vision providers, video domains, SQLAlchemy persistence, benchmarks, and research evidence tooling. Those live in separate distributions.

## Install

```powershell
pip install zeromodel
```

Runtime dependency: NumPy.

## Artifact Example

```python
from zeromodel.core import LayoutRecipe, ScoreTable, build_vpm

table = ScoreTable(
    values=[[0.9, 0.1], [0.4, 0.8]],
    row_ids=["candidate-a", "candidate-b"],
    metric_ids=["quality", "risk"],
)
recipe = LayoutRecipe.from_dict(
    {
        "version": "vpm-layout/0",
        "name": "quality-first",
        "row_order": {
            "kind": "lexicographic",
            "keys": [{"metric_id": "quality", "direction": "desc"}],
            "tie_break": "row_id",
        },
        "column_order": {"kind": "source"},
        "normalization": {"kind": "per_metric_minmax", "clip": True},
    }
)
artifact = build_vpm(table, recipe)
print(artifact.artifact_id)
```

## Policy Lookup Example

```python
from zeromodel.core import VPMPolicyLookup

reader = VPMPolicyLookup(artifact, action_metric_ids=["quality", "risk"])
decision = reader.read("candidate-a")
print(decision.action, decision.value)
```

Use `zeromodel.analysis`, `zeromodel.observation`, `zeromodel.vision`, `zeromodel.video`, and `zeromodel.persistence.sqlalchemy` for the higher-level packages.
