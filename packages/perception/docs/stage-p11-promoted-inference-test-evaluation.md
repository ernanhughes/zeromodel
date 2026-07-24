# Stage P11 — Promoted Inference and Final Test Evaluation

P11 executes the exact candidate and rejection threshold frozen by P10 through one promoted inference contract.

The runtime dispatches according to `PromotedPerceptionModelDTO.model_kind`, verifies the supplied translator identity and temporal-window identity when applicable, applies the frozen validation threshold, and returns an immutable accepted or rejected result with model, calibration, decision, input, score, margin, and action provenance.

Final evaluation runs the unchanged operating point on a caller-declared untouched `test` split and reports raw accuracy, accepted-only accuracy, coverage, mean margin, accepted and rejected counts, and per-interaction results.

P11 does not recalibrate on test data, change the promoted candidate, retrain either translator, infer that rejected examples are incorrect, or claim deployment performance beyond the evaluated test artifact.
