# Stage P16 — Operational Drift and Health Diagnosis

P16 freezes an `OperationalReferenceProfileDTO` from the exact untouched P11 test evaluation associated with a promoted model, then compares immutable P14/P15 production sequence windows against that reference.

The diagnosis keeps five signals separate: coverage drop, mean-margin drop, selected-action distribution distance, raw labeled-accuracy drop, and accepted-only labeled-accuracy drop. Action drift uses total-variation distance over the union of reference and production action labels.

`OperationalDriftPolicyDTO` declares every threshold and the minimum labeled-outcome count required for accuracy findings. When the label requirement is not met, accuracy is reported as `insufficient_evidence`; missing labels are never treated as errors and rejected predictions are never automatically treated as incorrect.

`OperationalHealthReportDTO` preserves the frozen reference identity, P14 metrics-report identity, inclusive production window, exact inference and outcome identities, production action distribution, per-metric findings, and deterministic overall status.

P16 is diagnostic only. It does not mutate the promoted model, recalibrate thresholds, alter lifecycle state, emit external alerts, or automatically trigger rollback.