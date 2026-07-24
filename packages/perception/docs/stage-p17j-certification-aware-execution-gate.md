# Stage P17J — Certification-Aware Execution Gate

## Objective

Use the P17I four-store certification audit before and after a fresh governed rollback.

```text
lifecycle + governance + attempt journal + certification ledger
    ↓
P17I preflight audit
    ↓
P17H governed rollback and certification
    ↓
P17I postflight audit
    ↓
CertificationExecutionGateDTO
```

## Preflight rule

Fresh work starts only when the four-store report is `valid`.

Both `attention_required` and `invalid` block the operation. Warning-level evidence gaps are not silently waived.

## Recovery boundary

The existing P17G and P17H paths remain responsible for the exact deterministic operation already in progress. P17J governs entry into fresh work and does not treat unrelated certification warnings as recovery authorization.

## Postflight rule

After certification, P17J requires:

- the exact certification bundle to be readable from the certification ledger;
- the restored bundle to equal the completed bundle;
- the new four-store report to be `valid`.

## Gate artifact

`CertificationExecutionGateDTO` binds the certification, attempt, receipt, preflight report, postflight report, and resulting pointer revision.

## Authority boundary

P17J adds no new model-state authority. The lifecycle store remains authoritative for the active model, while the other stores retain their existing evidence ownership.
