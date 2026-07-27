# P18H Durable Activation and Governed Rollback

## Objective

P18H closes the P18G durability gap with a bounded, in-process SQLite reference store and governed rollback execution for admitted perception-state activations.

## Existing P18G boundary

P18G already audits staged inactive P18F change sets, binds an activation admission to exact pre-state and result identities, applies forward operations atomically, and stores an inactive inverse rollback plan. P18G did not provide restart-safe persistence or rollback execution.

## Durable store design

`SqlitePromotionActivationStore` implements the existing DTO-only activation store boundary. It persists complete canonical DTO JSON blobs for active state, activation bundles, rollback admissions, and rollback bundles, plus explicit operation ordinals for forward and inverse operation history. Schema version `1` is recorded and unsupported future schema versions are rejected.

## Activation persistence

Activation commits run inside one SQLite `BEGIN IMMEDIATE` transaction. The store reloads the active state inside the transaction, compares it with the expected DTO, applies the forward plan to private state, inserts the immutable activation bundle and operation ordinals, and replaces the active state before commit.

## Rollback request and policy

`PromotionRollbackRequestDTO` names one rollback plan, the exact expected active state, requester, and reason. `PromotionRollbackPolicyDTO` limits operation count and permitted operation kinds and can require a non-empty reason. These DTOs are governance inputs, not organizational authorization.

## Rollback audit

`audit_promotion_rollback` returns `admissible`, `blocked`, `not_applicable`, or `already_executed`. It verifies the plan exists, has executable status, the current active state exactly equals the plan's activated state, baseline identity and version match, operation ordering and identities validate, policy allows every operation, and the predicted inverse result equals the stored restore state.

## Rollback admission

`PromotionRollbackAdmissionDTO` binds the request, policy, audit, rollback plan, exact current state identity and revision, predicted restore state identity, baseline identities, and inverse operation identities.

## Atomic rollback

Rollback commits run inside one SQLite transaction. The store reloads the persisted admission, active state, activation owner, and rollback plan, rechecks compare-and-swap state identity and revision, replays the exact stored inverse operations in order, verifies the restored state against the admission, inserts a rollback receipt and bundle, and replaces the active state before commit.

## Compare-and-swap protection

Rollback rejects stale admissions when the persisted active state id, revision, baseline id, or baseline version differs from the admitted expected state. It does not rewind across later activations.

## Idempotency

Duplicate activation commits for the same exact bundle return without rewriting state. Repeated rollback commits for the same admission return the stored rollback bundle. A rollback plan cannot be applied twice.

## Restart recovery

On reopen, the SQLite store reconstructs DTOs through their constructors. Active state, activation receipts, rollback plans, rollback admissions, executed rollback receipts, and operation ordering remain readable across restart.

## Failure injection

Tests inject failures during inverse replay and verify the active state and rollback plan status remain unchanged and no durable rollback receipt is created.

## Corruption handling

Malformed JSON, invalid DTO identities, and malformed operation ordinals are rejected explicitly. The store does not repair corruption.

## Public API

P18H exposes the rollback request, policy, audit, admission, receipt, bundle, `audit_promotion_rollback`, `authorize_promotion_rollback`, `execute_promotion_rollback`, and `SqlitePromotionActivationStore`.

## Validation

Focused P18G/P18H tests cover in-memory and SQLite parity, restart recovery, rollback idempotency, stale admission rejection, failure injection, malformed JSON, malformed operation ordinals, and public API exports.

## Operational guarantees

Within a single process using SQLite, admitted active state, activation receipts, and rollback plans survive restart. Rollback executes only when the current active state exactly matches the stored activated state, and the exact stored inverse operation plan restores the exact stored state atomically.

## Non-guarantees

P18H is not distributed consensus, a production service, semantic safety proof, automatic safety recovery, cryptographic signing, enterprise authorization, multi-step historical rewind, or a claim that an admitted activation is safe.

## Release implication

P18H completes the P18A-P18H in-process operational chain needed before the planned `v1.1.0` release-candidate pass.
