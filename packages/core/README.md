# ZeroModel Core

`zeromodel` is the lightweight core distribution for ZeroModel 1.2.0. It owns the `zeromodel.core` import namespace and provides the deterministic artifact kernel beneath the Visual AI Computing package system.

Core includes deterministic VPM artifact construction, stable artifact and matrix identities, basic views, `.vpm` bundle serialization, lightweight PNG/SVG rendering, exact bounded policy lookup, Lua export for compiled policy plans, and domain-neutral decision adjudication.

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

## Decision Adjudication

Decision adjudication keeps state correctness separate from action correctness. It classifies one bounded decision attempt into exactly one of four outcomes:

- `exact`: accepted, exact state, expected action;
- `action_equivalent`: accepted, non-exact state, but the expected action still wins;
- `action_changing`: accepted, but the selected action differs from the expected action;
- `rejected`: the procedure did not return an accepted decision.

```python
from zeromodel.core import adjudicate_decision

result = adjudicate_decision(
    accepted=True,
    expected_state={"door": "closed", "alarm": False},
    resolved_state={"door": "open", "alarm": False},
    expected_action="CONTINUE",
    selected_action="CONTINUE",
)

assert result.outcome.value == "action_equivalent"
assert result.state_match is False
assert result.action_correct is True
```

This is an adjudication record, not a promotion or deployment state machine. Higher-level packages may use adjudication results as evidence for evaluation, promotion, rollback, or research decisions, while retaining their own domain-specific identities, evidence, and rejection reasons.

See [Decision Adjudication](../../docs/decision-adjudication.md) for the contract and integration boundary.

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
