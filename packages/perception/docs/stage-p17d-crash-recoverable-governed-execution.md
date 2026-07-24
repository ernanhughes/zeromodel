# Stage P17D — Crash-Recoverable Governed Execution

## Objective

Close the interruption gap between P17B lifecycle rollback and P17C execution-receipt persistence without creating a second active-model authority.

The governed path is:

```text
persisted recommendation
    ↓
persisted approved disposition
    ↓
exact reviewed lifecycle pre-state
    ↓
compatible rollback
    ↓
persisted execution receipt
```

## Failure window

P17C can persist a receipt after rollback, but these are separate store operations. A process may stop after the lifecycle store commits the rollback and before the governance store commits the receipt.

The rollback is then real, but the durable governance chain appears incomplete.

## P17D operation

`execute_or_reconcile_approved_rollback` requires the exact persisted recommendation and disposition.

It supports three states:

1. **Receipt already exists** — return it idempotently without lifecycle mutation.
2. **Reviewed pre-state is still active** — execute the governed compatible rollback and append its receipt.
3. **Exact next rollback state exists without a receipt** — prove the transition and append the missing receipt.

## Reconciliation proof

Recovery requires all of the following:

- pointer revision is exactly the reviewed revision plus one;
- resulting active model is the approved target;
- pointer references an existing lifecycle transition;
- transition kind is `rollback`;
- transition sequence is exactly the expected next revision;
- previous and next model identities match the recommendation;
- transition actor and reason match the operator disposition.

Any unrelated supersession, rollback, deactivation, additional revision, different actor, or different reason is rejected.

## Authority boundary

The lifecycle store remains the sole authority for active-model state. The governance ledger does not replay or reconstruct lifecycle mutations. It only records a receipt for an already-proven exact transition.

P17D does not claim distributed atomic transactions across arbitrary stores. It provides deterministic, restart-safe reconciliation for the single known crash window.
