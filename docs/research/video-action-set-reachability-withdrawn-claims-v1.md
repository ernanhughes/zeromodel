# Video Action-Set Reachability Withdrawn Claims v1

Document version: `zeromodel-video-action-set-reachability-withdrawn-claims/v1`

## Identity

- current main SHA: `db9c990`
- historical prospective-stack SHA: `cf6acc72fc23638f1eee4b4317681a8b620a3188`
- integration PR number: `#46`
- integration merge SHA: `db9c990`
- benchmark version: `zeromodel-video-action-set-reachability-benchmark/v1`
- evidence version: `zeromodel-video-complete-row-evidence/v1`
- runtime-amendment version: `zeromodel-video-prospective-phase-access/v1`

## Exact Withdrawn Claims

- `optimized provider path implemented`
- `runtime equivalence verified`
- `measured runtime speedup`
- `runtime blocked after genuine optimization`
- `providers verified`
- `lexical uniqueness not used`
- `tie safety verified`
- `final access verified through zero counters`
- `prospective instrument verified`

## Reason For Withdrawal

- `score_all_rows_optimized` delegates to `score_all_rows_reference`.
- Runtime equivalence compared one implementation with itself.
- Runtime profiling ran the cold reference path before the warm delegated path.
- Module-level caches contaminated the timing comparison.
- No mutation-sensitivity control existed.
- Verification artifacts contained assigned rather than measured fields.

## Current Scientific Boundary

No candidate-set, conformal, invalid-input, temporal-negative, reachability, final-performance, or production-readiness conclusion is supported by the prospective artifacts.

## Current Operational Boundary

No development, calibration, or architecture-selection materialization may proceed until the reference instrument is correct and the verifier has demonstrated failure against known-bad mutations.

## Unresolved Corrective Work

- optimized path delegates to reference
- cache identity uses `id(prototypes)`
- score-vector identity includes raw floats
- package modules import from `examples`
- placeholder strings masquerade as SHA-256 digests
- P3 development pairs are prototype/prototype
- lexical rows are treated as semantic winners
- complete-evidence validation is incomplete
- benchmark seed is non-causal
- final episode plan is not concretely sealed
- conflicting-action splice can produce a valid prototype
- controls are relabelled valid episodes
- impossible transitions are metadata-only
- reachability tile is not genuinely action-conditioned
- verification fields are assigned as constants
- tests monkeypatch decisive checks into success
