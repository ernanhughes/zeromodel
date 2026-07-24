# Stage P17C — Durable Governance Ledger

## Objective

Persist the P17 recommendation and P17B operator-review chain without making the governance database a second lifecycle authority.

## Stored artifacts

`SqlitePerceptionGovernanceLedgerStore` stores three immutable artifact types:

1. `OperationalRecommendationDTO` — the evidence-owned, non-mutating recommendation;
2. `OperationalRecommendationDispositionDTO` — the final operator approval or rejection;
3. `GovernanceExecutionReceiptDTO` — the lifecycle transition and resulting pointer produced by an approved execution.

Each artifact is stored as canonical JSON under its content identity and reconstructed as the original DTO after restart.

## Ordering rules

The store enforces the governance sequence:

```text
persisted recommendation
    ↓
one final disposition
    ↓
zero or one execution receipt
```

A disposition cannot be persisted before its recommendation. An execution receipt cannot be persisted before an approved disposition.

## Replay protection

- Re-appending the exact same artifact is idempotent.
- A recommendation may have only one final disposition.
- An approved disposition may have only one execution receipt.
- Conflicting content under an existing identity is rejected.
- A second execution receipt for the same recommendation or disposition is rejected.

These rules prevent approval rewriting and execution replay after process restart.

## Authority boundary

The governance ledger does not activate, supersede, deactivate, or roll back models. The lifecycle store remains the sole authority for active-model state.

An execution receipt records a lifecycle result that has already occurred; it does not cause that result.

## Schema compatibility

The SQLite schema carries the explicit version:

```text
perception-sql-governance-schema/1
```

Unknown schema versions are rejected rather than interpreted optimistically.

## Deliberate boundary

P17C does not add automatic execution, scheduled review, multi-party approval, cryptographic signatures, or distributed transaction coordination between the lifecycle and governance databases.
