# Stage P17K — Durable Execution Admission

## Objective

Persist the complete P17J execution-admission proof so it survives process restart.

```text
P17I valid preflight
    ↓
P17J certification-aware execution gate
    ↓
P17I valid postflight
    ↓
CertificationExecutionAdmissionBundleDTO
    ↓
SqliteCertificationExecutionAdmissionStore
```

## Persisted evidence

Each immutable admission bundle contains:

- the P17J `CertificationExecutionGateDTO`;
- the exact P17I preflight report used to authorize fresh execution;
- the exact P17I postflight report that certified the completed state.

The bundle rejects mismatched report identities and requires both reports to be `valid`.

## Append-only rules

The store permits one admission proof per certification and deterministic attempt.

- an exact duplicate is idempotent;
- a conflicting gate or report bundle is rejected;
- records are canonically enumerable and fully reconstructable after restart;
- schema version mismatch is rejected explicitly.

## Execution entry point

`execute_and_persist_certification_admission(...)` captures the real four-store preflight, delegates execution and certification to P17J, proves the returned gate references that same preflight, persists the complete bundle, restores it, and verifies exact equality.

## Authority boundary

The admission ledger stores evidence only. It cannot mutate lifecycle state, approve recommendations, create receipts, or replace the governance, attempt, or certification stores.

## Exclusions

P17K does not:

- waive P17I warnings;
- repair invalid history;
- retry failed attempts;
- create a second lifecycle authority;
- infer or synthesize missing preflight evidence.
