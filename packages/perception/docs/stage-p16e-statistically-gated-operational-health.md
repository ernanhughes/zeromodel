# Stage P16E — Statistically Gated Operational Health

P16E repairs the operational-health blocker identified by external adversarial review.
A production window may not be classified as healthy or drifted merely because a
point estimate crosses a fixed threshold.

## Evidence gates

The public `OperationalDriftPolicyDTO` now requires explicit minimums for:

- reference examples;
- production inferences;
- labeled outcomes;
- labeled accepted inferences;
- production label coverage.

By default, accuracy comparison requires complete production labeling because the
frozen test reference is fully labeled. A selectively or partially labeled production
cohort therefore produces `insufficient_evidence`, not an accuracy conclusion.

## Window integrity

Health diagnosis requires one active-model pointer revision by default. A window that
spans activation, supersession, or rollback revisions is not treated as one homogeneous
operational cohort.

## Status precedence

Underpowered metrics are first-class `insufficient_evidence` findings. An underpowered
window cannot become `drifted` from one action-frequency observation and cannot become
`healthy` from sparse evidence.

Once every required metric is adequately supported, threshold crossings may produce
`drifted`; otherwise all supported metrics remaining within policy produce `healthy`.

## Compatibility

The immutable P16 reference, finding, and report DTO formats remain unchanged. The
public policy advances to `perception-operational-drift-policy/2`, and the package-level
`diagnose_operational_health` entry point now uses the gated implementation.

## Deliberate boundary

P16E still uses operational tolerance thresholds and total-variation distance. It does
not claim a formal hypothesis test, confidence interval, sequential-testing correction,
or causal diagnosis. Those may be added only when a concrete monitoring requirement
justifies them. P17 recommendations remain paused pending rollback compatibility and a
full vertical integration test.
