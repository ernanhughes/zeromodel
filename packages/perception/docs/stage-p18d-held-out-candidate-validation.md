# Stage P18D — Held-Out Candidate Validation

## Objective

Translate P18C missing-component candidates into explicit transition expectations and test them against a separately identified cohort of immutable P18A transition evidence.

```text
P18C TransitionDiscoveryReportDTO
    +
held-out P18A transition observations
    +
CandidateValidationPolicyDTO
    ↓
CandidateValidationReportDTO
```

P18D is the boundary between discovery and validation. A recurrent pattern from P18C remains an unvalidated association until it survives evidence that was not used to discover it.

## Held-out observations

`HeldOutTransitionObservationDTO` materializes one validation interaction from a P18A transition artifact. It stores the interaction identity, validation cohort identity, field schema, transition identity, and complete field evidence.

Distinct validation interactions may produce identical deterministic transition artifacts. They remain separate observations through their interaction and observation identities.

## Mandatory separation

P18D rejects validation when any of the following overlap with the discovery report:

- cohort identity;
- interaction identity;
- transition-evidence identity;
- observation identity.

The validation observations must also use the discovery field schema and belong to one validation cohort.

This is stricter than checking only a cohort label. Reusing the exact transition artifact under a different interaction name still constitutes evidence leakage and is rejected.

## Derived expectations

For each P18C candidate, P18D locates the exact source recurrence statistic and creates a content-addressed `CandidateValidationExpectationDTO`.

The expectation preserves:

- candidate and source-statistic identities;
- candidate kind and exact field identities;
- proposed change direction from P18C;
- minimum mean absolute change;
- minimum changed-value fraction;
- minimum signed-change magnitude for directional candidates.

Magnitude thresholds are derived from the discovery statistic through `minimum_magnitude_retention_fraction`. This makes the transfer rule explicit and reproducible rather than silently inventing a validation threshold.

## Per-interaction findings

Every candidate is evaluated against every held-out observation. Findings remain distinct:

- `confirmed`;
- `missing_change`;
- `insufficient_change`;
- `wrong_change_direction`;
- `inconclusive_direction`.

A multi-field signature is evaluated through the same value-count-weighted aggregation used by the discovery statistic.

## Candidate outcomes

`CandidateValidationResultDTO` records complete findings and derives one of four outcomes:

- `validated` — the confirmation fraction reaches policy;
- `rejected` — the rejection fraction reaches policy;
- `inconclusive` — evidence is sufficient but neither threshold is reached;
- `insufficient_validation_evidence` — the validation cohort is too small.

Findings are retained even when the result is evidence-insufficient. This keeps partial evidence visible without overstating it.

## Report outcomes

The complete report is classified as:

- `all_validated`;
- `mixed_outcomes`;
- `none_validated`;
- `insufficient_evidence`.

Rejected and inconclusive candidates remain first-class results. P18D never silently drops a failed discovery hypothesis.

## Integrity boundary

The held-out observation, validation policy, derived expectation, per-interaction finding, candidate result, and complete report are content-addressed.

P18D rejects candidate/statistic disagreement, unknown candidate fields, mixed schemas, mixed validation cohorts, duplicate interaction identities, discovery leakage, invalid policy thresholds, and identity tampering.

## Non-claims

A validated candidate means only that its field-level transition hypothesis reproduced under the declared held-out cohort and policy. It does not establish an object label, semantic relation, action cause, or universal generalization.

## Next stage

P18E should materialize a governed candidate-promotion proposal. Only P18D-validated candidates should be eligible, and promotion should remain reviewable rather than automatically rewriting the annotation or expectation model.
