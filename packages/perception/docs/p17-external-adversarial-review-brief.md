# ZeroModel Perception P17 — External Adversarial Review Brief

## Review purpose

P17 is no longer an implementation sketch. It is a complete governed rollback chain with durable evidence across lifecycle, recommendation, operator disposition, execution attempts, receipts, integrity audits, certifications, and final admission proof.

This review should determine whether the architecture is genuinely safer and more inspectable than a simpler transaction-oriented design, or whether the sequence has accumulated redundant evidence layers that create new failure modes.

Do not assume the design is correct because each individual DTO and store is internally consistent. Review the composition, crash boundaries, authority model, and operational usability as one system.

## Current stage

The package is at `PERCEPTION_STAGE = "P17K"`.

```text
production evidence and health report
    ↓
P17A evidence-owned recommendation
    ↓
P17B explicit operator disposition
    ↓
P17C durable governance ledger and execution receipt
    ↓
P17D crash-recoverable governed rollback
    ↓
P17E append-only execution-attempt journal
    ↓
P17F lifecycle/governance/attempt integrity audit
    ↓
P17G audit-gated rollback execution
    ↓
P17H durable gate certification
    ↓
P17I four-store certification integrity audit
    ↓
P17J certification-aware fresh-execution gate
    ↓
P17K durable execution-admission proof
```

## Store authorities

The architecture currently separates authority as follows:

| Store | Owned authority |
|---|---|
| lifecycle store | active model pointer, registered models, lifecycle transitions |
| governance ledger | recommendations, operator dispositions, execution receipts |
| attempt journal | prepared and terminal execution-attempt events |
| certification ledger | P17G gate plus exact surrounding P17F audit reports |
| admission ledger | P17J gate plus exact surrounding P17I reports |

Only the lifecycle store may mutate the active model. The other stores record governance or evidence.

## Safety properties claimed by the implementation

The implementation intends to provide these properties:

1. No rollback without an evidence-owned recommendation and explicit approved disposition.
2. Recommendation, disposition, compatibility assessment, lifecycle pointer, and target identities are immutable and content-addressed.
3. A crash after lifecycle mutation but before receipt persistence can be reconciled from exact lifecycle evidence.
4. Attempt intent is durable before lifecycle mutation.
5. Repeated calls are idempotent and cannot create a second rollback transition.
6. Cross-store contradictions are surfaced by deterministic read-only audits.
7. Fresh execution is blocked unless the complete persisted history is valid.
8. Exact preflight and postflight evidence survives restart.
9. Evidence stores cannot independently mutate model state.

Treat these as claims to falsify, not conclusions.

## Known implementation and process concerns

### 1. Cross-database atomicity remains absent

The design uses reconciliation rather than a distributed transaction. Review whether every interruption boundary is actually observable and recoverable, especially:

- lifecycle committed, attempt terminal event absent;
- receipt committed, certification absent;
- certification committed, admission absent;
- process crash after a returned success but before the caller receives it;
- concurrent workers operating on the same approved disposition.

### 2. Layer accumulation may be excessive

P17G through P17K add gate, certification, four-store audit, fresh-execution gate, and durable admission proof. Determine whether these are distinct necessary artifacts or repeated representations of the same successful transition.

Ask whether a smaller append-only transaction log or one unified governance aggregate could provide the same guarantees with fewer identities and databases.

### 3. Audits validate references more than semantics

The auditors strongly validate ownership and identity agreement. Review whether they prove the underlying semantic claims, including:

- compatibility really permits the rollback;
- the selected target is operationally safe;
- the health evidence remains applicable at execution time;
- the operator decision has not expired;
- the resulting model is serving successfully rather than merely becoming active.

### 4. Recovery and fresh-execution entry points overlap

P17D, P17E, P17G, P17H, P17J, and P17K expose progressively stronger execution functions. Review whether callers can accidentally select a weaker entry point and bypass later safeguards.

Determine whether only one production entry point should remain public, with lower-level functions explicitly marked internal or recovery-only.

### 5. Public API size and compatibility layering

The package root now exports the complete P17K surface through a compatibility layer preserving the prior P17F exports. Review whether this surface is coherent or whether operational internals should move behind a facade.

### 6. SQLite concurrency assumptions

Review transaction modes, uniqueness constraints, lock behavior, and race handling under multiple processes. Pay particular attention to check-then-insert sequences and whether SQLite exceptions are translated into deterministic domain errors.

### 7. Content-addressing is not automatically tamper evidence

A digest proves that a payload maps to an identity; it does not prove trusted origin or prevent a database administrator from replacing a row and recomputing its digest. Review whether signatures, chained hashes, trusted checkpoints, or external anchoring are required for the stated threat model.

### 8. Operational repair is intentionally missing

The system detects invalid and incomplete chains but generally refuses to repair them. Review whether operators have enough information and supported procedures to recover without manually editing databases.

## Required adversarial scenarios

Attempt to construct concrete failures for each scenario:

1. Two workers execute the same approved disposition concurrently.
2. Two workers execute different approved dispositions based on the same active pointer revision.
3. Lifecycle rollback commits and every later store write fails permanently.
4. An attempt is prepared, the target contract changes, and recovery occurs later.
5. A valid certification exists, but its exact preflight was generated from already stale recommendation evidence.
6. A receipt references a valid rollback transition that was produced outside the governed entry point.
7. A malicious or buggy caller invokes P17D directly after P17K is deployed.
8. An operator approves one assessment but a structurally similar alternative target is executed.
9. Database files are individually restored from backups taken at different times.
10. A later lifecycle transition makes an older admission historically valid but operationally misleading.
11. SQLite uniqueness races raise raw storage exceptions instead of stable domain errors.
12. A digest-bearing payload is recomputed after unauthorized modification.

For each scenario, state whether the current implementation:

- prevents it;
- detects it before mutation;
- detects it only afterward;
- can reconcile it automatically;
- requires operator intervention;
- cannot distinguish it from a valid history.

## Questions the review must answer

1. Is the authority split correct, or should governance, attempts, certifications, and admissions be one aggregate and one database transaction?
2. Which P17 artifacts are essential evidence, and which are redundant derived projections?
3. Is exact crash recovery actually proven for every write boundary?
4. Can callers bypass the strongest gate by using lower-level public functions?
5. Are current audits sufficient to claim integrity, or only referential consistency?
6. Does the architecture have a clear production facade and operator workflow?
7. What is the minimum safe P18, if any?
8. Should work pause for simplification before adding new capabilities?

## Requested review output

Return findings ranked by severity:

- blocker;
- high;
- medium;
- low;
- positive distinction.

For every blocker or high-severity finding, include:

- the exact invariant that fails;
- a concrete execution or crash sequence;
- the affected modules and stores;
- whether existing tests would catch it;
- the smallest corrective design;
- whether the correction belongs before P18.

End with one of these recommendations:

1. **Proceed to P18** — P17 is sufficiently coherent and safe.
2. **Consolidate first** — guarantees are sound, but the public surface or store model should be simplified.
3. **Repair first** — a correctness or recovery gap remains.
4. **Redesign the governance aggregate** — the current layering cannot be made reliable through incremental fixes.
