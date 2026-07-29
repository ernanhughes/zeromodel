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

## Stage O3.2 - Rebuildable Observation Graph

Stage O3.2 materializes a deterministic graph from immutable ledger evidence:

```text
transition ledger
        ↓
state-grouping recipe
        ↓
actual-observation assignments
        ↓
state classes
        ↓
action-labelled graph
```

The ledger remains the source of truth. The graph is a disposable materialized
view that can be deleted and rebuilt from the ledger snapshot, exact ledger
entries, grouping recipe, observation schema, and declared rule sets. Graph
nodes are equivalence classes under `ObserverStateGroupingRecipeDTO`, not
observation IDs; every exact observation artifact assigned to a node remains
listed as evidence. Grouping semantics are intentionally separate from
transition-comparison semantics.

Graph nodes represent actual observed states. Predicted observations are kept as
transition evidence and are not counted as traversed world states. Edges are
labelled by action and aggregate traversal count, verification-status counts,
supporting ledger entries, and predictor/environment rule-set IDs. The graph is
recipe-relative: the same ledger can produce a different graph under a different
grouping recipe.

If ledger integrity, semantic replay, or grouping/schema prerequisites fail, the
build result is canonical failure evidence with `graph=None`; no node, edge, or
assignment artifacts are emitted from invalid source evidence.

This stage does not add an event bus, graph database, habit promotion, policy
activation, graph-based action selection, embeddings, causal inference, or any
claim about retrieval or invocation savings.

```python
from zeromodel.observer import (
    ObserverGroupingFeatureDTO,
    ObserverStateGroupingRecipeDTO,
    build_observer_fixture_comparison_recipe,
    build_observer_observation_graph,
    verify_observer_graph_rebuild,
)

grouping = ObserverStateGroupingRecipeDTO.create(
    observation_schema_id=schema.schema_id,
    missing_feature_policy="separate_class",
    type_mismatch_policy="separate_class",
    feature_groupings=(
        ObserverGroupingFeatureDTO.create(
            feature_key="hidden.cooldown_remaining",
            mode="ignored",
        ),
        ObserverGroupingFeatureDTO.create(
            feature_key="visible.action_effect",
            mode="categorical",
        ),
        ObserverGroupingFeatureDTO.create(
            feature_key="visible.agent_x",
            mode="numeric_bucket",
            bucket_size=2.0,
        ),
        ObserverGroupingFeatureDTO.create(
            feature_key="visible.target_x",
            mode="exact",
        ),
    ),
)

build = build_observer_observation_graph(
    ledger_snapshot=episode.ledger_snapshot,
    entries=entries,
    grouping_recipe=grouping,
    observation_schema=schema,
    comparison_recipe=build_observer_fixture_comparison_recipe(schema),
    predictor_rule_sets=(rule_1,),
    environment_rule_sets=(rule_1, rule_2),
)

verification = verify_observer_graph_rebuild(
    expected_graph=build.graph,
    ledger_snapshot=episode.ledger_snapshot,
    entries=entries,
    grouping_recipe=grouping,
    observation_schema=schema,
    comparison_recipe=build_observer_fixture_comparison_recipe(schema),
    predictor_rule_sets=(rule_1,),
    environment_rule_sets=(rule_1, rule_2),
)
```

## Stage O3.3 - Promotion-Candidate Evidence

Stage O3.3 derives bounded analytical evidence from a successful observation
graph and its immutable ledger source:

```text
ledger
    ↓
observation graph
    ↓
novelty and recurrence
    ↓
stability and structural independence
    ↓
promotion candidate
```

Promotion candidates are content-addressed evidence snapshots under a declared
`ObserverPromotionEvidenceRecipeDTO`. They do not execute, mutate the graph,
alter policy, construct habits, or activate shortcuts. Recurrence means a
declared transition pattern appeared again; it is not correctness. Episode and
rule-regime diversity are structural independence proxies only, not statistical
independence claims. Rule-change survival is scenario-bound evidence, and
eligibility means only that a future compiler may inspect the candidate.

```python
from zeromodel.observer import (
    ObserverPromotionEvidenceRecipeDTO,
    analyze_observer_promotion_candidates,
)

promotion_recipe = ObserverPromotionEvidenceRecipeDTO.create(
    observation_graph_id=graph_build.graph.observation_graph_id,
    grouping_recipe_id=grouping_recipe.grouping_recipe_id,
    minimum_traversal_count=2,
    minimum_confirmed_count=2,
    minimum_independent_episode_count=1,
    maximum_contradicted_count=0,
    minimum_confirmation_ratio_numerator=1,
    minimum_confirmation_ratio_denominator=1,
)

analysis = analyze_observer_promotion_candidates(
    ledger_snapshot=episode.ledger_snapshot,
    entries=entries,
    graph_build=graph_build,
    grouping_recipe=grouping_recipe,
    promotion_recipe=promotion_recipe,
    observation_schema=schema,
)
assert analysis.status == "built"
```

## Stage O3.4 - Habit Compilation and Shadow Execution

Stage O3.4 converts one eligible promotion candidate into a bounded,
inspectable habit specification and evaluates it without control authority:

```text
eligible promotion candidate
        ↓
bounded habit specification
        ↓
counterexample guards
        ↓
shadow evaluation
        ↓
admission-review eligibility
```

The habit is inactive. Shadow recommendations are recorded beside the
authoritative action path, but the fixture still executes only the supplied
authoritative actions. Source matching is recomputed through the exact grouping
recipe, guards are derived from evidence, and known indistinguishable
counterexamples block compilation rather than becoming invented runtime
conditions. Agreement with authoritative actions is agreement only, not proof of
optimality; expected-target agreement is scenario-bound. Admission and
activation remain future stages.

The fixture shadow helper runs the authoritative fixture episode first, records
the actions that actually reached the ledger, and then performs deterministic
post-hoc shadow replay over that resulting evidence. It is not an activation
path and does not evaluate early enough to control the fixture.

```python
from zeromodel.observer import (
    ObserverHabitCompilationRecipeDTO,
    compile_observer_habit_specification,
    evaluate_observer_habit_over_ledger,
)

compilation_recipe = ObserverHabitCompilationRecipeDTO.create(
    promotion_recipe_id=promotion_recipe.promotion_recipe_id,
    grouping_recipe_id=grouping_recipe.grouping_recipe_id,
    observation_schema_id=schema.schema_id,
    allowed_guard_feature_keys=("visible.agent_x", "visible.target_x"),
    required_guard_feature_keys=(),
    forbidden_guard_feature_keys=(),
    maximum_guard_count=4,
    maximum_counterexample_guard_count=2,
    allow_exact_guards=True,
    allow_categorical_guards=True,
    allow_numeric_range_guards=True,
    require_counterexample_guards=False,
)

compiled = compile_observer_habit_specification(
    promotion_analysis=analysis,
    promotion_candidate=analysis.promotion_candidates[0],
    graph_build=graph_build,
    grouping_recipe=grouping_recipe,
    observation_schema=schema,
    compilation_recipe=compilation_recipe,
    ledger_snapshot=episode.ledger_snapshot,
    entries=entries,
)
habit = compiled.habit_specification
if habit is not None:
    shadow = evaluate_observer_habit_over_ledger(
        habit_specification=habit,
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=graph_build,
        grouping_recipe=grouping_recipe,
        observation_schema=schema,
    )
```

## Stage O3.5 - Habit Admission and Controlled Activation

Stage O3.5 introduces the first bounded path where an admitted habit may
influence a fixture action:

```text
shadow-audited habit
        ↓
admission
        ↓
inactive registry entry
        ↓
atomic activation
        ↓
habit or fallback decision
        ↓
post-action verification
        ↓
suspension or rollback
```

Admission is not activation. Admission decisions are immutable and recompute
thresholds from the exact shadow replay evidence named by the audit. Rejected
or malformed evidence does not create a registry entry. Admitted habits enter
the in-memory registry as `admitted_inactive`.

Activation is fixture-bound and uses an expected source registry snapshot as a
compare-and-swap token. Stage O3.5 permits one active habit per activation
scope. Every registry status change creates an append-only event and a new
immutable snapshot; rollback also creates new history instead of deleting old
history.

Active decisions always preserve the authoritative fallback action. A habit
recommendation executes only when the active registry entry, activation scope,
schema, grouping recipe, source state class, and guards all match. Abstention,
invalid evaluation, ambiguity, suspension, or retired state uses the fallback
path. Post-action transition verification is mandatory for active fixture
execution. Wrong active outcomes can suspend the habit automatically according
to the runtime safety recipe. Registry replay semantically reapplies every
event to reconstruct snapshots instead of trusting snapshot references alone.
This is a bounded fixture runtime, not a production policy engine, and it makes
no empirical performance claim.

```python
from zeromodel.observer import (
    InMemoryObserverHabitRegistry,
    ObserverHabitActivationRequestDTO,
    ObserverHabitActivationScopeDTO,
    activate_observer_habit,
    admit_observer_habit,
)

decision = admit_observer_habit(
    habit_specification=habit,
    shadow_audit=audit,
    historical_shadow_replay=historical_shadow,
    live_shadow_episodes=(live_shadow_episode,),
    admission_recipe=admission_recipe,
)

registry = InMemoryObserverHabitRegistry()
if decision.decision == "admit":
    registry.register_admission(
        habit_specification=habit,
        admission_decision=decision,
    )
    snapshot = registry.current_snapshot()
    request = ObserverHabitActivationRequestDTO.create(
        habit_specification_id=habit.habit_specification_id,
        expected_source_registry_snapshot_id=snapshot.habit_registry_snapshot_id,
        activation_scope_id=activation_scope.habit_activation_scope_id,
        reason_codes=("operator_activation",),
    )
    activation = activate_observer_habit(
        registry=registry,
        activation_scope=activation_scope,
        activation_request=request,
        habit_specification=habit,
    )
```

## Stage O3.6 - Durable Habit Registry

Stage O3.6 adds a SQLite-backed implementation of the same habit registry
contract used by `InMemoryObserverHabitRegistry`:

```text
in-memory registry semantics
    -> shared event reducer
    -> SQLite Store transaction
    -> event + snapshot + entries + head
    -> restart
    -> semantic recovery and verification
```

SQLite is a durable Store implementation, not a separate semantic authority.
The Store boundary remains DTO-only: public Store methods accept and return
registry DTOs or primitive query parameters, never SQLite rows or cursors.
Registry event, entry, and snapshot IDs remain content addressed through the
existing canonical JSON identity rules.

Every durable transition appends one immutable event, one immutable snapshot,
that snapshot's immutable entry projection rows, and a guarded current-head
update in a single SQLite transaction. Activation uses the source registry
snapshot ID as the compare-and-swap token, so stale writers fail without
persisting duplicate activation evidence. Current reads load the head and
referenced snapshot in a read transaction and validate canonical payloads
against relational projections.

The durable registry supports one SQLite writer at a time. Concurrent readers
observe committed snapshots only; lock contention can return the bounded
`database_locked` Store disposition. Restart derives active, suspended, retired,
generation, and rollback state from persisted registry history. Recovery is
explicit and conservative: it may repair an unambiguous missing head pointer
after semantic replay, but contradictory event or snapshot evidence is not
repaired automatically.

No distributed consensus, network coordination, background activation, or
performance claim is made. The strongest supported claim is that registry state
and activation decisions survive process restart and remain transactionally
consistent under the tested SQLite scenarios.

```python
from zeromodel.observer import SqliteObserverHabitRegistry

registry = SqliteObserverHabitRegistry.open("observer-habits.sqlite")
registry.register_admission(
    habit_specification=habit,
    admission_decision=admission_decision,
)
source = registry.current_snapshot()
registry.activate(
    habit_specification_id=habit.habit_specification_id,
    expected_source_registry_snapshot_id=source.habit_registry_snapshot_id,
)
assert registry.verify_integrity().status == "verified"
registry.close()
```

## Design Position

The package is for the first experiment described by the Observer design note:
detect one bounded transition contradiction, identify the affected policy
region, and activate a replacement policy with deterministic lineage. Broader
habit promotion, schema revision, and comparative System A-D evaluation remain
future work.
