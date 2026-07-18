# Video Stage Three Operational Contract

Date: July 18, 2026

Status: operationalization of the frozen preregistration without changing it

Parent preregistration: [video-stage-three-preregistration.md](/C:/Projects/zeromodel/docs/research/video-stage-three-preregistration.md)

Stage 2 parent commit: `d00e18b67fbe2f62617cd0ac47c7ee2f63487cb8`

## Scope

This document operationalizes the preregistered Stage 3 terms so implementation, architecture selection, calibration, and evaluation all use the same semantics.

It does not change:

- research questions
- architecture grid
- materiality rules
- kill conditions
- supported claim categories

## V4 evidence states

V4 uses exactly three semantic outcomes:

- `exact_row_accepted`
- `candidate_set_available`
- `no_sufficient_evidence`

### `exact_row_accepted`

Current-frame evidence uniquely supports one row at the frozen operating point.

### `candidate_set_available`

Current-frame evidence supports a bounded explicit set of rows, but not one unique exact row.

This is abstention from exact addressing.

### `no_sufficient_evidence`

Current-frame evidence does not support a governed candidate set.

Temporal logic may not convert this state into an accepted row.

## Candidate-set utility

Stage 3 reports these separately:

- raw truth-in-set recall
- bounded candidate-set recall
- unique-action candidate-set recall
- mixed-action candidate-set recall
- exact-row coverage

Candidate-set availability is never reported as exact-row coverage.

Promoted useful candidate sets are frozen to:

- `maximum useful candidate-set size = 3`

Candidate sets larger than 3 may be reported diagnostically but do not count as promoted candidate-set utility.

## Temporal state

For V5 version 1:

- temporal state advances only from an exact accepted row
- ambiguous V4 candidate sets do not advance state
- rejected frames do not advance state
- invalid ordering does not advance state
- identity mismatch does not advance state
- after an undeclared gap, state becomes unusable
- after a declared gap, only explicitly declared gap semantics apply
- no multi-hypothesis temporal state propagation is permitted

## Architecture D gateway

Architecture D is eligible only when all of the following hold on architecture-selection data:

1. Architecture A demonstrates nonzero safe utility.
2. Architecture B or C produces at least one additional correct exact acceptance or truth-containing useful candidate set not produced by A.
3. That added recovery introduces no distinguishable false accept, no conflicting-action exact accept, and no critical-contradiction exact accept.
4. The additional mechanism addresses a failure mode distinct from corrected visibility alone.

If any condition fails, D is not evaluated.

## Conservative tie-break ordering

When two operating points are otherwise tied, prefer in this order:

1. stronger minimum evidence mass
2. lower allowed contradiction
3. lower allowed critical contradiction
4. stronger exact-winner margin
5. smaller maximum candidate-set size
6. smaller candidate-set relative margin where that is conservative
7. more independent supporting regions
8. smaller registration bounds
9. fewer exact acceptances under a complete remaining tie
10. deterministic lexical ordering over serialized calibration values

## V5 safety invariants

V5 must never:

- inject a row absent from the V4 candidate set
- accept after `no_sufficient_evidence`
- carry forward a prior row
- carry forward a prior action
- update temporal state from an ambiguous candidate set
- silently cross an undeclared gap

## Registration tie-break intent

Stage 3 registration selection must prefer semantically stronger equal-distance matches in this order:

1. greater available informative-pixel mass
2. greater valid pixel count
3. greater geometric overlap
4. smaller Manhattan translation magnitude
5. smaller absolute vertical shift
6. smaller absolute horizontal shift
7. deterministic signed offset ordering

## Output mapping

Stage 3 evidence and metrics must preserve this distinction:

- exact acceptance is an exact addressing result
- candidate-set availability is bounded abstaining evidence
- no-sufficient-evidence is rejection

No later reporting step may merge these states.
