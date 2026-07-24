# Stage P10 — Calibration and Model Promotion

P10 calibrates single-frame and temporal rejection thresholds from a P9 validation comparison, then deterministically promotes one immutable candidate.

Calibration maximizes accepted accuracy subject to explicit minimum coverage, using coverage and the lower threshold as deterministic tie-breaks. Promotion compares calibrated accepted accuracy, raw accuracy, coverage, and finally model simplicity according to an explicit policy.

The promoted model records the selected candidate identity, calibrated rejection threshold, calibration identity, promotion-decision identity, validation comparison identity, training split, evaluation split, and temporal window identity when applicable.

P10 does not retrain candidates, use the test split for operating decisions, claim that the selected model is universally optimal, or erase the rejected candidate and its evidence.
