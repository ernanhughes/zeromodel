# Decision Adjudication

ZeroModel decision adjudication is a small core contract for keeping **state correctness** separate from **action correctness**.

A system can select the expected action while resolving the wrong state. Ordinary action accuracy collapses that case into success. Decision adjudication preserves the distinction explicitly.

## Contract

For one bounded decision attempt, core records:

```text
accepted
state_match
action_match
outcome
```

`outcome` is mutually exclusive:

| Outcome | Accepted | State match | Action match | Meaning |
|---|---:|---:|---:|---|
| `exact` | yes | yes | yes | The state and action both match. |
| `action_equivalent` | yes | no | yes | The state is non-exact, but the same action results. |
| `action_changing` | yes | either | no | The accepted result changes the expected action. |
| `rejected` | no | false | false | No decision was accepted. |

For accepted decisions, action mismatch takes precedence over state match. An exact state paired with the wrong action is therefore `action_changing`, not `exact`.

For rejected decisions, correctness fields are canonicalized to `False`. Placeholder values cannot turn a rejected attempt into a successful adjudication.

## API

```python
from zeromodel.core import (
    DecisionAdjudicationOutcome,
    adjudicate_decision,
)

result = adjudicate_decision(
    accepted=True,
    expected_state={"mode": "auto", "door": "closed"},
    resolved_state={"mode": "auto", "door": "open"},
    expected_action="CONTINUE",
    selected_action="CONTINUE",
)

assert result.outcome is DecisionAdjudicationOutcome.ACTION_EQUIVALENT
assert result.state_match is False
assert result.action_match is True
```

The helper deliberately uses caller-supplied equality. Core does not interpret state schemas, observations, policy semantics, confidence, evidence, or rejection reasons.

## Architectural boundary

Decision adjudication answers:

> **What happened on this decision attempt?**

It does **not** answer:

> **What lifecycle state should this provider or artifact enter?**

That second question belongs to higher-level evaluation and governance logic.

The intended relationship is:

```text
observation / evidence / provider output
                ↓
       bounded decision attempt
                ↓
       DecisionAdjudication
                ↓
      metrics / research evidence
                ↓
 optional higher-level lifecycle policy
```

The core abstraction should therefore remain free of provider IDs, artifact references, confidence thresholds, promotion rules, and database concerns. Owning packages bind those richer records around the adjudication result.

## Existing integration

The video action-set provider evaluation path already used the same four-way semantics before this core abstraction existed. The integration extracts those semantics into `zeromodel.core` and keeps the existing provider-specific case DTO responsible for:

- expected and predicted structured state;
- policy and provider identities;
- expected and predicted decision traces;
- per-factor state matches;
- provider confidence and latency;
- rejection reason;
- response evidence and metadata;
- deterministic case identity.

The serialized outcome strings remain:

```text
exact
action_equivalent
action_changing
rejected
```

so historical summary mathematics and stored evidence remain compatible.

## Why this distinction matters

Suppose a provider evaluates 1,000 observations and selects the expected action 940 times. Action accuracy alone reports 94%.

Decision adjudication can reveal whether those 940 cases were exact-state successes or action-equivalent errors. For example:

```text
exact               680
action_equivalent   260
action_changing      40
rejected              20
```

Both views say 940 actions were correct, but only the adjudicated view exposes that 260 accepted decisions were right at the action layer without resolving the exact state.

The taxonomy does not by itself prove that the resolved state is semantically meaningful, that the expected action is safe, or that a policy is correct. It preserves a bounded distinction that downstream evaluation can measure instead of hiding it inside one accuracy number.

## Extension rule

Do not make every ZeroModel subsystem emit this taxonomy merely for architectural symmetry.

A subsystem should use decision adjudication only when all of the following exist:

1. a declared expected/reference state or decision context;
2. a resolved or predicted state/context that can be compared under a declared equality contract;
3. an expected/reference action or consequence;
4. a selected action or explicit rejection.

If Search, Critic, Observer, or another package cannot satisfy those semantics without distorting its domain, it should not adopt this abstraction.
