# Video Stage Three V4 Provider

Date: July 18, 2026

Status: frame-local V4 provider contract and implementation note

Parent documents:

- [video-stage-three-preregistration.md](/C:/Projects/zeromodel/docs/research/video-stage-three-preregistration.md)
- [video-stage-three-operational-contract.md](/C:/Projects/zeromodel/docs/research/video-stage-three-operational-contract.md)
- [video-stage-three-evidence-mechanics.md](/C:/Projects/zeromodel/docs/research/video-stage-three-evidence-mechanics.md)

## Scope

This document records the frame-local V4 provider contract implemented in Stage 3.

It defines unit-tested provider behavior for architectures `A`, `B`, and `C`.

It does not establish:

- architecture selection
- feasible operating points on the preregistered benchmark
- final calibration
- final evaluation
- any V5 temporal behavior

V4 unit tests demonstrate contract behaviour on constructed fixtures. They do not establish safe nonzero coverage on the preregistered Stage 3 benchmark.

## Provider boundary

`DiscriminativeEvidenceProvider` is strictly current-frame only.

It accepts:

- prototypes
- masks
- regions
- calibration
- policy artifact identity
- source scope

It does not accept:

- previous row
- previous action
- temporal state
- clip history

## Architecture semantics

All architectures emit a higher-is-better normalized `candidate_strength`.

Shared ranking direction:

1. greater `candidate_strength`
2. greater available informative mass
3. greater available informative fraction
4. lower contradiction
5. lower critical contradiction
6. greater supporting-region count
7. lexical row ID
8. lexical prototype observation ID

### Architecture A

Purpose:

- test corrected visibility without importing the full support-versus-contradiction logic as a decision rule

Current raw score:

- Stage 2-style weighted regional distance reconstructed from region evidence

Current `candidate_strength`:

- `(1 - raw_distance) * available_informative_fraction`

Active gates:

- minimum available mass
- minimum available fraction
- candidate relative margin
- conflicting-action separation
- exact winner threshold
- exact winner margin
- maximum candidate-set size

Inactive gates:

- minimum support
- maximum contradiction
- maximum critical contradiction
- minimum supporting regions

### Architecture B

Purpose:

- test discriminative weighting and available-mass renormalization

Current raw score:

- support ratio over available informative mass

Current `candidate_strength`:

- support ratio

Active gates:

- Architecture A gates
- minimum support

Inactive gates:

- maximum contradiction
- maximum critical contradiction
- minimum supporting regions

### Architecture C

Purpose:

- preserve support, contradiction, critical contradiction, and conflicting-action evidence separately

Current raw score:

- `support_ratio - contradiction_ratio - critical_ratio`, clipped at zero

Current `candidate_strength`:

- the clipped score above

Active gates:

- minimum available mass
- minimum available fraction
- minimum support
- maximum contradiction
- maximum critical contradiction
- candidate relative margin
- conflicting-action separation
- minimum supporting regions
- exact winner threshold
- exact winner margin
- maximum candidate-set size

Inactive gates:

- none

## Candidate-set inclusion

A row is eligible for candidate-set inclusion only when all active candidate-set gates pass independently for that row.

The provider never silently truncates oversized candidate sets.

When more than `maximum_candidate_set_size` rows qualify, the result is:

- `no_sufficient_evidence`
- rejection reason: `candidate_set_too_large`

## Exact acceptance

The provider emits `exact_row_accepted` only when:

1. exactly one row is eligible for exact acceptance
2. that row is top-ranked
3. it is also candidate-set eligible

The mapped `VisualAddressDecision` is accepted only in this state.

## Candidate-set availability

When one or more rows independently pass candidate-set gates but exact uniqueness is not established, the provider emits:

- `candidate_set_available`

The mapped `VisualAddressDecision` remains:

- `accepted=False`
- `matched_row_id=None`

This preserves the operational distinction between exact addressing and bounded abstaining evidence.

## Rejection

The provider emits `no_sufficient_evidence` when:

- no row passes candidate-set gates
- the candidate set would be too large
- geometry or identity validation fails
- architecture-specific evidence is undefined

## Cache layers

The implementation currently uses:

- a raw candidate cache keyed by observation identity and pre-calibration evidence identity
- a calibrated decision cache keyed by raw-cache identity plus calibration digest

The cache key includes:

- observation raw digest
- observation source ID
- observation geometry
- prototype collection digest
- mask payload digest
- region-spec digest
- registration contract digest set
- architecture ID
- evidence mechanics version
- calibration digest
- policy artifact ID
- source scope

## VisualAddressDecision mapping

`exact_row_accepted`:

- `accepted=True`
- `matched_row_id=<winner>`

`candidate_set_available`:

- `accepted=False`
- `matched_row_id=None`
- reason preserved as `candidate_set_available`

`no_sufficient_evidence`:

- `accepted=False`
- `matched_row_id=None`
- typed rejection reason preserved

## What this block does not establish

This block does not prove:

- that any architecture is feasible on the preregistered Stage 3 benchmark
- that any calibration is safe
- that candidate-set utility is material
- that V5 temporal narrowing is valid
- that any empirical claim is warranted
