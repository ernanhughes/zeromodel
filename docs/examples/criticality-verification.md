# Criticality-aware policy verification

ZeroModel 1.0.11 extends bounded policy artifacts with two separate evidence metrics and an exhaustive finite property checker.

- `criticality` is `max(action values) - min(action values)`. Describe it as VIPER-style criticality only when the source columns carry Q-values or an equivalent consequence-bearing teacher signal.
- `decision_margin` is `best action value - second-best action value`.

The two metrics answer different questions:

```text
criticality:
How costly could a poor action be?

decision margin:
How decisively does the winning action beat its nearest alternative?
```

## Run the complete fixture

```bash
python examples/criticality_verification.py \
  --output-dir docs/assets/criticality-verification
```

The example builds:

1. a Q-bearing 112-state arcade policy;
2. a source-order policy artifact;
3. a criticality-first view of the same scored source;
4. a passing verification artifact;
5. a deliberately corrupted policy in which `FIRE` wins while unaligned;
6. a failed verification artifact containing the exact counterexample;
7. a repaired policy with a new identity;
8. a passing re-verification artifact linked to the repaired policy.

Generated outputs include `.vpm` bundles, PNG views, and `criticality_verification_results.json`.

## Add diagnostic evidence

```python
from zeromodel import ScoreTable, with_q_diagnostics

ACTIONS = ("LEFT", "RIGHT", "STAY", "FIRE")

source = ScoreTable(
    values=[
        [-1.0, -1.0, 0.0, 5.0],
        [-1.0, 3.0, 0.5, -2.0],
    ],
    row_ids=[
        "tank=3|target=3|cooldown=0",
        "tank=0|target=1|cooldown=0",
    ],
    metric_ids=ACTIONS,
)

enriched = with_q_diagnostics(
    source,
    action_metric_ids=ACTIONS,
)
```

The new source metrics are:

```text
LEFT
RIGHT
STAY
FIRE
criticality
decision_margin
```

## Keep evidence out of action selection

```python
from zeromodel import VPMPolicyLookup

reader = VPMPolicyLookup(
    artifact,
    action_metric_ids=ACTIONS,
    evidence_metric_ids=(
        "criticality",
        "decision_margin",
    ),
)

decision = reader.read(
    "tank=3|target=3|cooldown=0"
)

print(decision.action)
print(decision.candidates)
print(decision.evidence)
```

`evidence_metric_ids` are returned with the decision but never participate in the argmax.

## Declare a finite policy property

```python
from zeromodel import PolicyPropertySpec

fire_requires_alignment = PolicyPropertySpec.from_dict({
    "id": "fire_requires_alignment_and_ready",
    "version": "1",
    "assert": {
        "implies": [
            {"eq": [{"var": "winner"}, "FIRE"]},
            {"all": [
                {
                    "eq": [
                        {"var": "state.tank"},
                        {"var": "state.target"},
                    ]
                },
                {"eq": [{"var": "state.cooldown"}, 0]},
            ]},
        ]
    },
})
```

The first checker version supports `key=value|...` row IDs and the operators:

```text
var
eq
ne
lt
lte
gt
gte
in
all
any
not
implies
```

## Check every declared row

```python
from zeromodel import PolicyPropertyChecker

checker = PolicyPropertyChecker(
    artifact,
    action_metric_ids=ACTIONS,
    evidence_metric_ids=(
        "criticality",
        "decision_margin",
    ),
)

report = checker.check([
    fire_requires_alignment,
])

verification_artifact = report.to_vpm()
```

The report records:

- the exact policy artifact ID;
- checker version;
- property-spec digest;
- rows checked;
- pass or fail;
- exact counterexample rows and selected cells;
- action candidates and evidence metrics.

The resulting verification artifact carries a provenance parent with relation `verifies` pointing to the exact policy artifact checked.

## Boundary

This is exhaustive checking of declarative row-level properties over a finite compiled policy. It is not general formal verification of continuous dynamics, temporal safety, liveness, authorship, authorization, or universal policy correctness.

## Research lineage

The design is inspired by VIPER:

> Osbert Bastani, Yewen Pu, and Armando Solar-Lezama, “Verifiable Reinforcement Learning via Policy Extraction,” NeurIPS 2018.

VIPER extracts decision-tree policies from neural teachers so that properties can be verified on a more tractable representation. ZeroModel extends that direction by preserving criticality and candidate evidence in an identity-bearing artifact, then linking verification results and counterexamples to the exact artifact checked.
