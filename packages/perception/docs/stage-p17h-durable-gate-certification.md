# Stage P17H — Durable Audit-Gate Certification

## Objective

Persist the complete P17G execution certification so it survives process restart and can be inspected without reconstructing historical pre-execution state.

```text
P17F pre-audit
    ↓
P17G audit-gated rollback or recovery
    ↓
P17F valid post-audit
    ↓
GovernanceExecutionGateDTO
    + exact pre-audit report
    + exact post-audit report
        ↓
GovernanceExecutionCertificationBundleDTO
        ↓
SqliteGovernanceCertificationStore
```

## Why the bundle is necessary

P17G content-addresses the identities of the pre- and post-audit reports, but its return value is process-local. After restart, the resulting gate identity alone cannot reproduce the exact pre-execution report because lifecycle state has already changed.

P17H therefore stores all four immutable artifacts together:

- `GovernanceExecutionCertificationDTO`;
- `GovernanceExecutionGateDTO`;
- exact `GovernanceIntegrityAuditReportDTO` before execution;
- exact valid `GovernanceIntegrityAuditReportDTO` after execution.

## Certification invariants

A certification bundle is accepted only when:

- the certification references the supplied gate;
- recommendation, disposition, attempt, receipt, and audit identities agree;
- the gate references the supplied pre- and post-audits;
- the post-audit status is `valid`;
- the post-audit active pointer revision equals the gate result revision.

## Append-only storage

`SqliteGovernanceCertificationStore` permits one certification per:

- gate;
- attempt;
- operator disposition.

An identical re-append is idempotent. A conflicting certification for an already certified attempt or disposition is rejected rather than overwritten.

The store persists the full JSON payloads and reconstructs nested audit findings and related identities after restart.

## Execution entry point

`execute_and_certify_audit_gated_rollback(...)`:

1. captures the deterministic pre-audit;
2. delegates execution and recovery to P17G;
3. verifies that the gate references the same pre-audit;
4. builds the certification bundle;
5. appends the bundle to the certification store;
6. returns the durable bundle, attempt, and receipt.

## Authority boundary

The certification database is evidence, not lifecycle state. It cannot register, activate, supersede, deactivate, or roll back a model.

The lifecycle store remains the sole active-model authority. The governance ledger remains the authority for recommendations, dispositions, and receipts. The attempt journal remains the authority for execution intent and terminal attempt outcome.

## Deliberate boundary

P17H does not yet make certifications an input to the P17F cross-store integrity audit. That is a separate integration step because it changes the auditor from three-store reconciliation to four-store reconciliation and requires explicit legacy behavior for gates created before certification persistence existed.
