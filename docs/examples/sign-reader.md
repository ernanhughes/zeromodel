# Signs, not directions

ZeroModel 1.0 includes a small policy consumer for the simplest version of the idea:
compile a bounded policy space into a VPM artifact once, then use runtime state to
address that artifact and read the sign.

The blog can call this a **SignReader**.  The code API is `VPMPolicyLookup`.

```text
compile time:
  scored state/action table -> VPM artifact

runtime:
  state -> row_id -> VPMPolicyLookup -> LEFT / RIGHT / STAY / FIRE
```

This is not a claim that a VPM invents the policy at runtime.  The policy is
compiled beforehand.  The runtime consumer only reads a deterministic artifact.
That is the point: no model call, no planner, no chain of instructions.

## Minimal policy lookup

Rows are state descriptors.  Metric columns are actions.

```python
from zeromodel import LayoutRecipe, ScoreTable, VPMPolicyLookup, build_vpm

source = ScoreTable(
    values=[
        [1.0, 0.0, 0.0, 0.0],  # state says LEFT
        [0.0, 1.0, 0.0, 0.0],  # state says RIGHT
        [0.0, 0.0, 0.0, 1.0],  # state says FIRE
    ],
    row_ids=["tank=3|target=0|cooldown=0", "tank=3|target=6|cooldown=0", "tank=3|target=3|cooldown=0"],
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
reader = VPMPolicyLookup(artifact)
decision = reader.read("tank=3|target=3|cooldown=0")

assert decision.action == "FIRE"
assert decision.artifact_id == artifact.artifact_id
assert decision.row_id == "tank=3|target=3|cooldown=0"
assert decision.metric_id == "FIRE"
```

The returned decision contains the source row, source metric, view row, and view
column that produced the action, so a replay can cite the exact cell used.

## Tiny arcade shooter demo

The repository includes a deterministic headless arcade-shooter example:

```bash
python examples/arcade_shooter_policy.py
```

The example compiles a closed-world shooter policy into one VPM artifact.  At
runtime the game state becomes a row id, the reader returns the action sign, and
the game steps.  Tests assert that the artifact policy clears the wave, beats a
random-action baseline, and produces byte-identical action traces when the same
artifact is replayed.

## Honest boundary

This validates a narrow but important claim: a bounded policy can be compiled
into an addressable, inspectable, deterministic artifact and consumed without a
model at decision time.

It does not validate open-world reasoning, off-grid generalization, top-left
optimization, or manifold discovery.  Those are separate ZeroModel capabilities
and need their own fixtures and benchmarks.
