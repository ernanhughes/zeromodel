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
- `compare_observer_transition`
- `build_transition_record`
- `build_contradiction_artifact`
- `build_replacement_policy_artifact`
- `verify_observer_transition`

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

## Design Position

The package is for the first experiment described by the Observer design note:
detect one bounded transition contradiction, identify the affected policy
region, and activate a replacement policy with deterministic lineage. Broader
habit promotion, schema revision, and comparative System A-D evaluation remain
future work.
