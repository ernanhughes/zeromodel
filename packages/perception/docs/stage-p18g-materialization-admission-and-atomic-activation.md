# Stage P18G — Materialization Admission and Atomic Activation

## Objective

Admit and activate a non-empty P18F materialization change set only when the active annotation, relation, transition-expectation, schema, and model-version baseline remains exactly the baseline against which the change set was constructed.

```text
P18F staged inactive change set
    + active policy state
    + activation policy
    ↓
exact baseline audit
    ↓ admissible only
content-addressed admission
    ↓
compare-and-swap atomic forward application
    ↓
active state + receipt + stored inactive rollback plan
```

P18G is the first P18 stage that can change the active policy state. It therefore treats activation as a transaction rather than as a sequence of independent object writes.

## Active state

`ActivePromotionStateDTO` is a complete content-addressed state containing:

- monotonic revision;
- active baseline-version identity;
- field-schema identity;
- active region annotations;
- active relations;
- active transition expectations;
- the most recently activated change-set identity.

The DTO validates annotation and relation payload identities, relation-member integrity, transition-expectation targets, schema compatibility, uniqueness, ordering, and its own state identity.

Its `baseline()` method reconstructs the exact `PromotionMaterializationBaselineDTO` used by P18F.

## Admission policy

`PromotionActivationPolicyDTO` declares:

- the maximum number of materialized changes admitted in one activation;
- the permitted materialization target kinds.

Exact baseline equality, complete inverse coverage, and atomic application are invariants rather than optional policy settings.

## Pre-activation audit

`audit_promotion_activation(...)` compares the P18F change set with one exact active-state snapshot.

The audit verifies:

- the change set is non-empty and `staged_inactive`;
- the field schema is unchanged;
- the active baseline-version identity equals the staged baseline version;
- the complete active baseline identity equals the staged baseline identity;
- policy limits permit every staged target;
- every forward operation resolves to one exact materialized DTO payload;
- no operation overwrites an active object;
- relations reference active annotations;
- transition expectations reference active targets;
- the complete resulting state can be constructed and content-addressed.

Audit outcomes are:

- `admissible`;
- `blocked`;
- `not_applicable` for an explicit P18F `no_approved_changes` result.

A blocked or not-applicable report cannot contain a proposed resulting state.

## Admission

`authorize_promotion_activation(...)` converts only an admissible audit into a `PromotionActivationAdmissionDTO`.

The admission binds:

- change set and policy;
- audit report;
- exact expected state and baseline;
- exact predicted resulting state and baseline;
- ordered forward operation identities;
- ordered inverse operation identities.

If any supplied state differs from the audited state, authorization fails.

## Atomic store boundary

`PromotionActivationStore` is a DTO-only compare-and-swap boundary:

```python
get_active_state()
commit_activation(expected_state, change_set, bundle)
get_activation_bundle(change_set_id)
list_activation_bundles()
```

`InMemoryPromotionActivationStore` is the P18G reference implementation. It:

1. acquires one store lock;
2. verifies the current state still equals the admitted expected state;
3. replays every forward operation into private copied maps;
4. validates the actual result against the admitted resulting state;
5. prepares the activation and rollback ledger update;
6. swaps the active state and ledger together.

No active object or ledger artifact is changed before the final swap. A failure during any operation leaves the previous state and activation ledger unchanged.

The compare-and-swap check prevents a valid audit from being reused after another writer changes the active state.

## Activation artifacts

A successful execution persists one `PromotionActivationBundleDTO` containing:

- admissible audit report;
- admission;
- activation receipt;
- exact resulting active state;
- inactive rollback plan.

The receipt binds the previous and resulting state, baseline, model version, revision, and forward operation sequence.

The rollback plan stores:

- the complete restore state;
- activated and restore baseline identities;
- activated and restore baseline-version identities;
- the exact reverse-ordered P18F inverse operations.

The plan remains `stored_inactive`. P18G does not execute rollback.

## Baseline version advance

The resulting baseline-version identity is derived deterministically from:

- previous state identity;
- previous baseline identity and version;
- P18F change-set identity;
- next revision.

A successful activation therefore cannot retain the old baseline version or depend on wall-clock time.

## Atomicity tests

The P18G suite exercises the complete P18C → P18D → P18E → P18F → P18G chain and verifies:

- exact baseline admission and activation;
- active annotation/relation/expectation insertion;
- receipt and rollback lineage;
- baseline drift rejection;
- target-kind policy rejection;
- compare-and-swap race rejection;
- injected failure after an intermediate operation with no partial state or ledger mutation;
- explicit handling of no-approved-change sets;
- identity tamper rejection across every new contract.

## Operational boundary

P18G does not:

- infer ontology;
- alter a P18F change set;
- admit against a merely similar baseline;
- partially apply a forward plan;
- execute the inverse plan;
- claim that an activated component caused the transition evidence.

It activates the exact reviewed, validated, materialized policy additions—or activates nothing.

## Next stage

P18H should add restart-safe SQLite persistence for the P18G store protocol and separately governed rollback admission. Rollback must require the current active state to equal the rollback plan's exact activated state before applying every inverse operation atomically.
