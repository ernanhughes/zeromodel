# Stage P17I — Certification Integrity Audit

## Objective

Extend the P17F read-only governance reconciliation across the P17H certification ledger.

```text
lifecycle store
    + governance ledger
    + execution attempt journal
    + certification ledger
        ↓
CertificationIntegrityAuditReportDTO
```

## Composition

P17I first runs the existing three-store `audit_governance_integrity(...)`. It then validates every durable certification against the recommendation, disposition, attempt, receipt, and resulting pointer revision it claims to certify.

## Status model

- `valid`: the underlying governance audit is valid and certification ownership is complete;
- `attention_required`: no contradiction is proven, but a warning exists;
- `invalid`: the underlying governance audit is invalid or a certification has a missing or contradictory ownership link.

## Checks

The audit detects:

- certifications whose recommendation, disposition, attempt, or receipt is missing;
- attempt ownership that differs from the certification;
- disposition ownership that differs from the certification;
- receipt ownership that differs from the certification;
- receipt result revisions that differ from the certified revision;
- successful journaled attempts without a durable P17H certification;
- invalid or attention-required three-store governance history.

## Historical certification semantics

A certification is evidence about a specific historical transition. It does not become stale merely because later valid lifecycle transitions occur.

The auditor therefore compares a certification with its own receipt and ownership chain. It does not require the certified pointer revision to remain the current active revision.

## Missing certification classification

A successful attempt without a P17H bundle is reported as `successful_attempt_uncertified` with warning severity. This preserves compatibility with executions completed before certification persistence existed while making the evidence gap visible.

## Determinism and authority

Findings and reports are content-addressed and canonically ordered. The audit is read-only and cannot repair, certify, execute, approve, reject, or mutate lifecycle state.
