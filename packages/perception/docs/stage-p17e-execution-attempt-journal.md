# Stage P17E — Governed Execution Attempt Journal

## Objective

Make every governed rollback attempt durably visible before lifecycle mutation begins and preserve one append-only terminal outcome after execution, reconciliation, idempotent linkage, or governed failure.

P17D repairs the specific crash window where lifecycle rollback committed but its governance receipt did not. P17E adds the missing operational record that an attempt was started and whether it eventually terminated.

## Event model

```text
GovernedExecutionAttemptDTO
    ↓
prepared
    ↓
completed | reconciled | idempotent | failed
```

The attempt identity binds:

- recommendation identity;
- approved disposition identity;
- reviewed pointer identity and revision;
- reviewed active model;
- approved target model;
- current compatibility contract;
- target compatibility contract.

## Terminal meanings

- `completed`: the reviewed pre-state was still active, so this invocation executed the rollback.
- `reconciled`: a prepared-only attempt resumed after lifecycle rollback had committed without its terminal journal event.
- `idempotent`: the execution receipt already existed before this attempt was linked to it.
- `failed`: governed validation rejected the operation before a receipt could be established.

A failed attempt is terminal. Retrying requires a new recommendation and operator disposition rather than silently reopening rejected evidence.

## Storage boundary

`SqliteGovernedExecutionAttemptStore` is a separate append-only SQLite contract. This avoids silently changing the P17C governance-ledger schema in place.

The stores retain distinct authority:

- lifecycle store: active-model truth and transitions;
- governance ledger: recommendation, disposition, and receipt evidence;
- attempt journal: operational execution intent and terminal attempt outcome.

The attempt journal never changes lifecycle state directly.

## Restart rule

A process crash after `prepared` leaves a visible incomplete attempt. The next invocation resumes through P17D:

1. execute if the reviewed pre-state remains current;
2. reconcile if the exact approved rollback post-state already exists;
3. reject any unrelated lifecycle movement;
4. append exactly one terminal attempt event.

## Deliberate boundary

P17E does not add automatic retries, background workers, leases, timeouts, or multi-party approval. Those require explicit scheduling and concurrency contracts rather than inference from an unfinished attempt.
