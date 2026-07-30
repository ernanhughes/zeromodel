# ZeroModel Core

`zeromodel` is the lightweight core distribution for ZeroModel 1.2.0. It owns the `zeromodel.core` import namespace and provides the deterministic artifact kernel beneath the Visual AI Computing package system.

Core includes deterministic VPM artifact construction, stable artifact and matrix identities, basic views, `.vpm` bundle serialization, lightweight PNG/SVG rendering, exact bounded policy lookup, and Lua export for compiled policy plans.

Core deliberately excludes analysis, observation contracts, vision providers, perception, observer applications, video domains, SQLAlchemy persistence, artifact stores, trust, navigation, benchmarks, and research evidence tooling. Those live in separate distributions.

## Install

```powershell
pip install zeromodel==1.2.0
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

Use the owning namespaces directly for higher-level capabilities:

- `zeromodel.analysis`
- `zeromodel.observation`
- `zeromodel.vision`
- `zeromodel.perception`
- `zeromodel.observer`
- `zeromodel.video`
- `zeromodel.persistence.sqlalchemy`
- `zeromodel.artifacts`
- `zeromodel.trust`
- `zeromodel.navigation`

See the [system README](../../README.md) and [claims audit](../../docs/claims-audit.md).
