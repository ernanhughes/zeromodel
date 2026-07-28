# zeromodel-observer

`zeromodel-observer` is the bounded Observer demonstration package for
canonical perceptual artifacts, verified transitions, and auditable policy
replacement.

## Scope

This package does not claim a general Observer architecture, habit formation,
continual learning, universal state similarity, or open-world perception. Its
initial contract is narrower:

```text
policy artifact
  + predicted transition
  + observed transition
  + declared comparison recipe
  + contradiction artifact
  + replacement lineage
```

The supported claim is that a small perceptual action policy can record and
replay why a local transition mismatch produced a replacement artifact.

## Package Boundary

Observer is a consumer package. It depends on `core`, `observation`, and
`perception`; it must not widen the `zeromodel.core` artifact kernel.

## Public API

The top-level public surface is:

- `ObserverObservationArtifactDTO`
- `ObserverComparisonRecipeDTO`
- `ObserverComparisonResultDTO`
- `ObserverTransitionRecordDTO`
- `ObserverContradictionArtifactDTO`
- `ObserverReplacementPolicyArtifactDTO`
- `ObserverTransitionVerificationDTO`
- `ObserverTransitionVerificationError`
- `ObserverRepairConstraintDTO`
- `ObserverProposedChangeDTO`
- `ObserverRepairProposalDTO`
- `ObserverRepairProposalError`
- `compare_observer_transition`
- `build_transition_record`
- `build_contradiction_artifact`
- `build_replacement_policy_artifact`
- `verify_observer_transition`
- `propose_observer_repair`

## Stage O1 - Transition Verification

Stage O1 compares two already-created observation artifacts through a declared
recipe and returns one canonical verification result. It does not predict
observations, repair policies, persist records, call model providers, or
implement habits.

Feature keys are projected from observation artifacts with explicit namespaces:

```text
visible.agent_x
history.previous_action
hidden.cooldown
```

Missing required evidence is reported as `inconclusive`. It is not treated as a
successful match and does not create a contradiction artifact.

```python
from zeromodel.observer import (
    ObserverComparisonRecipeDTO,
    ObserverObservationArtifactDTO,
    verify_observer_transition,
)

recipe = ObserverComparisonRecipeDTO.create(
    observable_feature_keys=("visible.agent_x", "visible.target_x"),
    action_effect_keys=("visible.action_effect",),
    policy_consequence_key="visible.next_action",
    hidden_state_keys=("hidden.cooldown",),
)

predicted = ObserverObservationArtifactDTO.create(
    visible_state_features={
        "agent_x": 5,
        "target_x": 9,
        "action_effect": "moved_right",
        "next_action": "move_right",
    },
    hidden_state_uncertainty={"cooldown": "clear"},
    sequence_index=1,
)

observed = ObserverObservationArtifactDTO.create(
    visible_state_features={
        "agent_x": 5,
        "target_x": 9,
        "action_effect": "moved_right",
        "next_action": "move_right",
    },
    hidden_state_uncertainty={"cooldown": "clear"},
    sequence_index=1,
)

verification = verify_observer_transition(
    recipe=recipe,
    predicted_observation=predicted,
    observed_observation=observed,
    policy_artifact_id="policy:A",
    state_before_id="state:before",
    action="move_right",
    affected_policy_row_id="row:before",
    predicted_decision_margin=0.3,
    observed_decision_margin=0.3,
    hidden_state_hypotheses_remaining=1,
)

verification.verification_status
verification.comparison_result
verification.transition_record
verification.contradiction_artifact
```

## Stage O2 - Bounded Repair Proposal

Stage O2 converts a contradicted transition verification into a bounded repair
proposal. It validates that a caller-supplied proposal is well-formed,
traceable, and inside declared repair authority. It does not mutate a policy,
verify a repair, create a replacement artifact, persist anything, or activate
anything.

`repairable` means eligible for the next stage of candidate generation and
audit. It does not mean the proposed change is correct.
`requested_changes` records caller intent. `proposed_changes` contains only
changes executable under the current schema and constraints.

```python
from zeromodel.observer import (
    ObserverProposedChangeDTO,
    ObserverRepairConstraintDTO,
    propose_observer_repair,
)

constraint = ObserverRepairConstraintDTO.create(
    allowed_row_ids=("row:cooldown-sensitive",),
    allowed_cell_ids=("row:cooldown-sensitive/action:move_right",),
    allowed_context_keys=("hidden.cooldown",),
    max_changed_rows=1,
    max_changed_cells=1,
    allow_action_value_change=True,
    allow_new_context_precondition=True,
)

change = ObserverProposedChangeDTO.create(
    target_kind="policy_cell",
    target_id="row:cooldown-sensitive/action:move_right",
    operation="replace",
    field_name="action_value",
    old_value="move_right",
    proposed_value="wait",
    condition_keys=("hidden.cooldown",),
)

proposal = propose_observer_repair(
    verification=verification,
    constraint=constraint,
    available_policy_row_ids=("row:cooldown-sensitive",),
    available_policy_cell_ids=("row:cooldown-sensitive/action:move_right",),
    represented_context_keys=("hidden.cooldown",),
    requested_changes=(change,),
    rationale_codes=(
        "action_effect_mismatch",
        "affected_row_localised",
        "repair_scope_bounded",
    ),
)

proposal.disposition
proposal.affected_row_ids
proposal.required_context_keys
proposal.missing_schema_keys
proposal.requested_changes
proposal.proposed_changes
```

## Design Position

The package is for the first experiment described by the Observer design note:
detect one bounded transition contradiction, identify the affected policy
region, and activate a replacement policy with deterministic lineage. Broader
habit promotion, schema revision, and comparative System A-D evaluation remain
future work.
