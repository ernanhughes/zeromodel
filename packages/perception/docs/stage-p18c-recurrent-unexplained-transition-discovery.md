# Stage P18C — Recurrent Unexplained Transition Discovery

## Objective

Aggregate immutable P18A transition evidence and P18B conformance findings across a declared discovery cohort to identify recurrent unmarked fields and exact co-occurrence signatures.

```text
interaction identity
    + P18A TransitionEvidenceVPMDTO
    + P18B TransitionConformanceReportDTO
    ↓
TransitionDiscoveryObservationDTO
    ↓ many observations from one cohort
field recurrence + exact unexplained signatures
    ↓
TransitionDiscoveryReportDTO
    ├── complete recurrence statistics
    └── MissingComponentCandidateDTO hypotheses
```

P18C does not assign a semantic label to a field. It identifies spatial evidence that repeatedly escaped the current annotation and expectation model.

## Discovery observations

`TransitionDiscoveryObservationDTO.create(...)` verifies that the P18B report references the supplied P18A artifact and that every unexplained finding reproduces the exact P18A field measurements.

Observations with no unexplained findings remain in the cohort. They are required for an honest recurrence denominator.

Distinct interactions may reference identical transition or conformance artifacts. P18C counts interactions through their unique interaction and observation identities rather than rejecting repeated deterministic evidence.

## Recurrence statistics

P18C records two kinds of statistics:

- `field` — how often one exact unmarked field is unexplained;
- `cooccurrence_signature` — how often the same complete set of two or more unexplained fields appears together.

Each statistic preserves:

- observation and occurrence counts;
- recurrence fraction across the complete cohort;
- weighted mean absolute, signed, and changed-fraction measurements;
- positive, negative, and neutral direction counts;
- dominant direction and direction consistency;
- supporting interaction, observation, transition, conformance, and finding identities.

Exact signatures are intentionally conservative. An observation containing extra unexplained fields belongs to a different signature.

## Candidate policy

`TransitionDiscoveryPolicyDTO` separates overall evidence sufficiency from candidate thresholds:

- minimum cohort observation count;
- minimum field occurrence count and recurrence fraction;
- minimum signature occurrence count and recurrence fraction;
- minimum directional consistency;
- signed-change epsilon.

The report can therefore distinguish:

- `insufficient_evidence` — the cohort is too small to propose candidates;
- `no_candidates` — the cohort is sufficient but no statistic crosses policy;
- `candidates_found` — one or more statistics cross policy.

## Candidate hypotheses

A `MissingComponentCandidateDTO` is always marked `candidate_unvalidated`.

Where signed evidence is sufficiently consistent, the candidate proposes a future expectation of `increase` or `decrease`. Otherwise it proposes the non-directional expectation `change`.

The proposal is intended for a later, separate validation cohort. P18C does not validate its own discoveries on the same interactions that generated them.

## Integrity boundary

The observation, policy, recurrence statistics, candidates, and complete discovery report are content-addressed. P18C rejects mixed cohorts, mixed field schemas, duplicate interaction identities, lineage mismatches, metric mismatches, invalid thresholds, and identity tampering.

## Non-claims

P18C does not claim that:

- a recurrent field is an object;
- a co-occurrence signature is a relation;
- a candidate caused the action or state transition;
- recurrence in the discovery cohort generalizes;
- a proposed expectation is confirmed.

Its claim is narrower: the declared cohort contains recurrent, addressable unexplained transition evidence that is suitable for a new falsifiable hypothesis.

## Next stage

The next slice should validate P18C candidates against a separately identified evaluation cohort. Candidate promotion must require held-out evidence and preserve failed candidates rather than silently discarding them.
