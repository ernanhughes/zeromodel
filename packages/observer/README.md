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
- `ObserverObservationSchemaDTO`
- `ObserverFeatureDefinitionDTO`
- `ObserverComparisonRecipeDTO`
- `ObserverFeatureComparisonDTO`
- `ObserverFeatureComparisonResultDTO`
- `ObserverComparisonResultDTO`
- `ObserverHiddenStateHypothesisDTO`
- `ObserverHiddenStateHypothesisSetDTO`
- `ObserverPolicyConsequenceEvidenceDTO`
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

## Stage O3.0 - Contract Hardening

Stage O3.0 hardens the evidence contracts used by transition verification. It
still does not generate predictions, invoke an Observer, execute a wake policy,
create a graph, promote habits, or materialize policy repairs.

The hardened path is:

```text
observation schema
  + schema-validated observations
  + per-feature comparison specs
  + hidden-state hypothesis set
  + external policy consequence evidence
        ↓
versioned comparison result
        ↓
transition verification
```

Feature comparison semantics are declared per feature. `exact` requires strict
type identity, so `True` does not equal `1`, and `1` does not equal `1.0`.
`numeric_tolerance` requires explicit absolute and relative tolerances and
rejects booleans, NaN, and infinity.

Observation artifacts now carry an `observation_schema_id`. Construction rejects
undeclared keys, missing required keys, mistyped values, and non-finite numeric
values. Schema identity is part of observation identity; comparing observations
from different schemas is inconclusive unless a later stage declares
compatibility.

Hidden-state exhaustion is derived from an
`ObserverHiddenStateHypothesisSetDTO`. The public transition verifier no longer
accepts a naked caller-supplied remaining count.

Policy consequence equivalence is supplied as
`ObserverPolicyConsequenceEvidenceDTO`, with external decision-trace IDs. A
caller-authored feature such as `visible.next_action` is not proof of policy
reader equivalence.

```python
from zeromodel.observer import (
    ObserverComparisonRecipeDTO,
    ObserverFeatureComparisonDTO,
    ObserverFeatureDefinitionDTO,
    ObserverHiddenStateHypothesisDTO,
    ObserverHiddenStateHypothesisSetDTO,
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
    verify_observer_transition,
)

agent_x = ObserverFeatureComparisonDTO.create(
    feature_key="visible.agent_x",
    mode="numeric_tolerance",
    expected_type="number",
    absolute_tolerance=0.0,
    relative_tolerance=0.01,
)

schema = ObserverObservationSchemaDTO.create(
    schema_name="cooldown-v1",
    features=(
        ObserverFeatureDefinitionDTO.create(
            qualified_key="hidden.cooldown",
            value_type="str",
            required=False,
        ),
        ObserverFeatureDefinitionDTO.create(
            qualified_key="visible.agent_x",
            value_type="number",
            required=True,
            comparison_id=agent_x.comparison_id,
        ),
    ),
)

recipe = ObserverComparisonRecipeDTO.create(
    feature_comparisons=(agent_x,),
    observable_feature_keys=("visible.agent_x",),
)

predicted = ObserverObservationArtifactDTO.create(
    observation_schema=schema,
    visible_state_features={"agent_x": 5},
    hidden_state_uncertainty={"cooldown": "clear"},
    sequence_index=1,
)
observed = ObserverObservationArtifactDTO.create(
    observation_schema=schema,
    visible_state_features={"agent_x": 5},
    hidden_state_uncertainty={"cooldown": "clear"},
    sequence_index=1,
)

hypotheses = ObserverHiddenStateHypothesisSetDTO.create(
    observation_schema_id=schema.schema_id,
    hypotheses=(
        ObserverHiddenStateHypothesisDTO.create(
            state_key="hidden.cooldown",
            state_value="clear",
            status="possible",
        ),
    ),
)

verification = verify_observer_transition(
    recipe=recipe,
    predicted_observation=predicted,
    observed_observation=observed,
    policy_artifact_id="policy:A",
    state_before_id="state:before",
    action="move_right",
    affected_policy_row_id="row:before",
    hidden_state_hypothesis_set=hypotheses,
)
```

Migration note: `observer-comparison-recipe/1` is not silently reinterpreted.
Stage O3.0 callers must construct `observer-comparison-recipe/2` with explicit
`feature_comparisons`. `observer-observation-artifact/2` requires an
`ObserverObservationSchemaDTO`.

## Stage O3.1 - Deterministic Transition Ledger

Stage O3.1 adds the first executable, replayable Observer fixture loop:

```text
fixture state
        ↓
deterministic prediction
        ↓
actual fixture step
        ↓
transition verification
        ↓
append-only ledger
        ↓
wake-policy replay
```

The predictor is deterministic and fixture-specific. The actual environment
executor is separate, so tests can pin the predictor to `fixture-rule/1` while
the environment switches to `fixture-rule/2` mid-episode. The ledger is
currently in-memory and is the source of truth for integrity validation.
Semantic replay is separate: it reruns prediction, execution, and transition
verification from the recorded source state, action, and declared rule-set
identities. Wake policies are evaluated after the fact from stored evidence;
wake replay does not rerun the environment.

This stage does not create a graph, event bus, habit, repair candidate, policy
activation, persistence layer, or claim about invocation savings.

```python
from zeromodel.observer import (
    ObserverFixtureActionDTO,
    ObserverFixtureRuleScheduleEntryDTO,
    ObserverFixtureRuleSetDTO,
    ObserverFixtureStateDTO,
    ObserverWakePolicyDTO,
    build_observer_fixture_comparison_recipe,
    build_observer_fixture_observation_schema,
    evaluate_wake_policy_over_ledger,
    run_observer_fixture_episode,
)

schema = build_observer_fixture_observation_schema()
rule_1 = ObserverFixtureRuleSetDTO.create(
    fixture_id="fixture:line",
    rule_version="fixture-rule/1",
    minimum_position=0,
    maximum_position=6,
    cooldown_period=1,
    cooldown_effect="block",
    observation_schema_id=schema.schema_id,
)
rule_2 = ObserverFixtureRuleSetDTO.create(
    fixture_id="fixture:line",
    rule_version="fixture-rule/2",
    minimum_position=0,
    maximum_position=6,
    cooldown_period=1,
    cooldown_effect="reverse",
    observation_schema_id=schema.schema_id,
)
initial = ObserverFixtureStateDTO.create(
    fixture_id="fixture:line",
    rule_set_id=rule_1.fixture_rule_set_id,
    episode_id="episode:1",
    step_index=0,
    agent_x=1,
    target_x=6,
)

episode, entries = run_observer_fixture_episode(
    initial_state=initial,
    actions=(
        ObserverFixtureActionDTO.create(action_name="move_right"),
        ObserverFixtureActionDTO.create(action_name="move_right"),
    ),
    predictor_rule_set=rule_1,
    environment_rule_schedule=(
        ObserverFixtureRuleScheduleEntryDTO.create(
            start_step=0,
            rule_set_id=rule_1.fixture_rule_set_id,
        ),
        ObserverFixtureRuleScheduleEntryDTO.create(
            start_step=1,
            rule_set_id=rule_2.fixture_rule_set_id,
        ),
    ),
    environment_rule_sets=(rule_1, rule_2),
    observation_schema=schema,
    comparison_recipe=build_observer_fixture_comparison_recipe(schema),
)

policy = ObserverWakePolicyDTO.create(
    policy_name="contradiction-only",
    wake_on_contradiction=True,
)
replay = evaluate_wake_policy_over_ledger(
    ledger_snapshot=episode.ledger_snapshot,
    entries=entries,
    wake_policy=policy,
)
```

## Design Position

The package is for the first experiment described by the Observer design note:
detect one bounded transition contradiction, identify the affected policy
region, and activate a replacement policy with deterministic lineage. Broader
habit promotion, schema revision, and comparative System A-D evaluation remain
future work.
