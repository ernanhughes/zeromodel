# Stage P17F — Governance Integrity Audit

## Objective

Provide one deterministic, read-only reconciliation pass across the three operational authorities introduced through P17:

```text
lifecycle store
    +
governance ledger
    +
execution attempt journal
        ↓
GovernanceIntegrityAuditReportDTO
```

The auditor does not repair, delete, retry, approve, reject, activate, or roll back anything.

## Status model

- `valid`: no warning or error finding exists;
- `attention_required`: at least one recoverable or historical warning exists, but no contradiction is proven;
- `invalid`: at least one cross-store contradiction, orphan, or impossible success chain exists.

Informational findings do not downgrade a valid report.

## Findings

The audit checks:

- dispositions reference existing recommendations and preserve recommendation status;
- receipts belong to approved dispositions and the same recommendation;
- receipts reference real rollback transitions;
- receipt pointer revisions match transition sequence numbers;
- receipt target models match lifecycle rollback targets;
- attempts reference the exact recommendation and approved disposition;
- attempts begin with `prepared`;
- prepared-only attempts are surfaced as recoverable incomplete work;
- failed attempts do not also possess success receipts;
- successful terminal events resolve to exactly matching receipts;
- terminal pointer revisions and rollback targets match their receipts;
- receipts without P17E attempt records are identified as legacy, not automatically corrupt.

## Determinism

Every finding and report is content-addressed. Findings are canonically sorted, and the report identity includes:

- current pointer identity and revision;
- artifact counts;
- report status;
- ordered finding identities;
- audit semantics and version.

Repeating the audit over unchanged stores yields an identical report.

## Authority boundary

The audit reads DTOs from each store through public methods. `SqliteGovernedExecutionAttemptStore.list_attempts()` is added as a canonical read surface so the auditor does not reach into private SQLite state.

The lifecycle store remains the sole active-model authority. The audit report is evidence about integrity, not operational state.

## Deliberate boundary

P17F does not automatically resume prepared attempts, repair missing receipts, or quarantine invalid records. Those actions must remain explicit and separately governed.
