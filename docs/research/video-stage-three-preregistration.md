# Video Stage Three Preregistration

Date: July 18, 2026

Status: frozen before any Stage 3 fresh final split generation

Stage 2 parent commit: `d00e18b67fbe2f62617cd0ac47c7ee2f63487cb8`

Stage 2 benchmark digest: `sha256:589bb074e1b53b06657cfb75bf7b8d67eae43cc5f76e7237ab07f23ccca49c75`

Stage 2 split digest: `sha256:d25b694b3cce93bf93f58239163331f3f6370d32a2b5cce53b4541902b0f8c23`

## Boundary

Stage 2 evidence is known and immutable. Files under `docs/results/video-policy-reader-v1/` will not be rewritten, regenerated, or reused as Stage 3 final evaluation identities.

Stage 2 final evidence may be used only for post-hoc diagnosis. Stage 3 must use fresh identities, fresh calibration data, and a fresh final holdout.

## Research questions

### RQ1 — Visibility semantics

Does separating semantic evidence availability from geometric registration overlap create a safe, nonzero-coverage operating point?

### RQ2 — Discriminative evidence

Does scoring only pixels or features that discriminate candidate rows improve safe frame-local coverage?

### RQ3 — Action conflict

Does separately measuring evidence against conflicting-action rows prevent unsafe acceptance while preserving more benign coverage?

### RQ4 — Candidate sets

Can a frame safely support a set of plausible rows even when it cannot support a unique frame-local row?

### RQ5 — Temporal intersection

Can declared temporal admissibility reduce a current-frame candidate set to one row without introducing stale-state carry-forward?

### RQ6 — Material utility

Does V4 materially outperform frozen V2, and does V5 materially outperform V4, on an untouched fresh final split?

## New systems

- V4 — discriminative set-valued frame evidence
- V5 — V4 plus safe temporal candidate intersection

Historical systems V0 through V3 remain executable and frozen.

## Planned V4 contracts

Stage 3 will implement explicit immutable contracts for:

- discriminative region specification
- discriminative mask specification
- evidence availability specification
- candidate evidence
- candidate sets
- action-conflict evidence
- V4 calibration
- V4 selection artifact
- provider contract and evidence identity

Expected public concepts include names close to:

- `DiscriminativeRegionSpec`
- `DiscriminativeMaskSpec`
- `DiscriminativeEvidenceCalibration`
- `RegionDiscriminativeEvidence`
- `DiscriminativeRowCandidate`
- `DiscriminativeCandidateSet`
- `DiscriminativeEvidenceProvider`

## Architecture grid

The frozen architecture grid will compare exactly these frame-local variants:

### A — Corrected visibility only

Use the Stage 2 regional distance structure, but compute evidence availability over informative aligned pixels rather than minimum rectangular overlap.

### B — Discriminative weighted evidence

Aggregate only available discriminative evidence mass and renormalize over the available mass.

### C — Support plus contradiction

Track supportive and contradictory evidence separately, including critical contradiction and conflicting-action evidence.

### D — Combined architecture

Only if A through C justify it on the architecture-selection split, evaluate a combined architecture using corrected visibility, discriminative weighting, and separate support/contradiction accounting.

No additional architecture variants may be added after this preregistration without a new benchmark version.

## V4 candidate-set rule

V4 may emit three distinct outcomes:

- exact row accepted
- candidate set available
- no sufficient evidence

A row may enter the candidate set only when it satisfies frozen current-frame evidence rules over:

- minimum available discriminative mass
- maximum contradiction
- maximum critical contradiction
- candidate-set relative margin
- conflicting-action separation
- maximum candidate-set size

Candidate-set availability will not be reported as exact-row coverage.

## V5 temporal rule

V5 may accept an exact row only when:

1. the row is already present in the V4 candidate set
2. the row has sufficient current-frame evidence
3. the transition contract permits the row
4. the temporal intersection is a singleton
5. no critical contradiction exists
6. frame ordering and identity are valid
7. no undeclared gap is silently crossed
8. no prior row or action is copied forward

If the selected row is absent from the V4 candidate set, the result is invalid.

## Data protocol

Stage 3 will use distinct splits for:

- prototypes
- diagnostic development
- architecture selection
- benign threshold calibration
- rejection threshold calibration
- final benign evaluation
- final distinguishable negative evaluation
- information-theoretic controls

Fresh final identities will be generated after this preregistration using a deterministic seed derived from a fixed domain separator plus the preregistration commit SHA.

## Operating-point selection hierarchy

The calibration selection rule is frozen as:

1. zero distinguishable false accepts
2. zero conflicting-action accepts
3. zero critical-contradiction accepts
4. nonzero exact-row or candidate-set benign coverage
5. maximize benign exact-row coverage
6. maximize correct candidate-set coverage
7. maximize accepted exact-row accuracy
8. minimize candidate-set size
9. prefer the more conservative operating point
10. deterministic tie ordering

## Feasibility and materiality rules

### V4 feasibility

V4 is feasible only when calibration finds an operating point with:

- zero observed distinguishable false accepts
- zero accepted conflicting-action errors
- zero accepted critical contradictions
- nonzero benign current-frame utility

### V4 material improvement over frozen V2

V4 materially improves frozen V2 only when final evaluation shows:

1. zero new distinguishable false accepts
2. zero conflicting-action accepts
3. zero critical-contradiction accepts
4. at least 10 correct benign exact-row acceptances or at least 10% benign exact-row coverage
5. accepted exact-row precision of 100%
6. the gain is not limited to information-theoretic controls
7. no final-set tuning occurred

### V5 material improvement over V4

V5 materially improves V4 only when:

1. zero new distinguishable false accepts
2. zero new conflicting-action accepts
3. V5 never converts a correct V4 exact result into an incorrect accepted result
4. V5 resolves at least one V4 ambiguous candidate set correctly
5. every resolved row was already in the V4 candidate set
6. every resolved row had sufficient current-frame evidence
7. no gain depends on stale-state carry-forward
8. the paired gain is visible in raw counts
9. benign coverage loss, if any, is reported
10. undeclared gaps are never silently crossed

## Kill conditions

The Stage 3 kill conditions are frozen as:

- Kill A — corrected visibility has no utility
- Kill B — no feasible V4
- Kill C — candidate sets are too broad
- Kill D — V4 is feasible but not material
- Kill E — V5 does not materially improve V4
- Kill F — temporal candidate injection
- Kill G — stale-state dependence
- Kill H — benchmark leakage
- Kill I — complexity without utility

## Supported-claim templates

Allowed primary result categories:

- Category 1 — material V4 and material V5
- Category 2 — material V4, non-material V5
- Category 3 — candidate-set utility only
- Category 4 — feasible but non-material V4
- Category 5 — no feasible V4
- Category 6 — invalid measurement

## Prohibited claims

Stage 3 may not claim:

- general video understanding
- arbitrary-motion invariance
- object recognition
- semantic scene understanding
- real-world camera robustness
- semantic temporal reasoning
- learned world models
- useful approximate video coverage outside the frozen benchmark
- temporal improvement without paired evidence
- exact visual addressing from candidate-set recall
- perception when the result depends on state carry-forward

## Version rule

The package version remains `1.0.11`.

Stage 3 completion does not itself authorize a version bump.
