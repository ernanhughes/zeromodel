# Video Stage Three Benchmark And Selection

Date: July 18, 2026

Status: pre-final benchmark freeze, architecture selection, and calibration preparation

Parent documents:

- [video-stage-three-preregistration.md](/C:/Projects/zeromodel/docs/research/video-stage-three-preregistration.md)
- [video-stage-three-operational-contract.md](/C:/Projects/zeromodel/docs/research/video-stage-three-operational-contract.md)
- [video-stage-three-evidence-mechanics.md](/C:/Projects/zeromodel/docs/research/video-stage-three-evidence-mechanics.md)
- [video-stage-three-v4-provider.md](/C:/Projects/zeromodel/docs/research/video-stage-three-v4-provider.md)

## Scope

This block freezes the pre-final Stage 3 benchmark preparation artifacts.

It covers:

- lightweight Stage 2 diagnosis verification
- Stage 3 benchmark and split manifests
- frozen region and mask manifests
- frozen Architecture D semantics
- architecture-selection grid and evidence
- architecture D gateway
- selected-architecture artifact
- calibration status artifact
- pre-final verification

It does not cover:

- V5
- final split evaluation
- final metrics
- final claim adjudication

## Stage 2 diagnosis verification

The benchmark driver now separates:

- `--verify-stage2-diagnosis`
- `--rebuild-stage2-diagnosis`

`--verify-stage2-diagnosis` is read-only and checks the frozen committed diagnosis files against a committed digest manifest.

`--rebuild-stage2-diagnosis` remains the expensive regeneration path and may exceed the local runtime window.

## Stage 3 benchmark freeze

The benchmark contract remains:

- benchmark version: `zeromodel-video-discriminative-evidence-stage3/v1`
- preregistration commit: `e6d3c2461a3e7fc783026907aa1ab5b803c878f3`
- final seed material: `zeromodel-stage3-final-v1|e6d3c2461a3e7fc783026907aa1ab5b803c878f3`

The generated split roles are:

- `prototype`
- `diagnostic_development`
- `architecture_selection_benign`
- `architecture_selection_negative`
- `benign_calibration`
- `rejection_calibration`
- `final_benign`
- `final_distinguishable_negative`
- `information_theoretic_control`

The current implementation freezes final identities without running final prediction or writing final metrics.

## Regions and masks

The region manifest freezes:

- target evidence region
- cooldown evidence region
- tank evidence region

The mask manifest freezes:

- row-informative masks
- action-conflict masks
- stable masks
- separation weights
- derivation contract

Mask derivation uses only:

- prototype observations
- diagnostic-development observations

## Architecture D

Architecture D is now frozen before selection.

Its current candidate strength is the combined support-minus-contradiction form already used by the frame-local combined provider semantics:

- normalized support
- minus normalized contradiction
- minus normalized critical contradiction
- clipped at zero

No new unregistered mechanism was added for D.

## Architecture selection

The architecture-selection grid is persisted before evaluation.

The current bounded implementation uses a small deterministic grid to remain computationally tractable in the local environment.

The current selection artifact records:

- `selection_status`
- all-results digest
- architecture-selection split digest
- grid digest
- mask digest
- region digest
- policy artifact identity
- source scope
- simplicity order

On the current frozen preparation run, the recorded status is:

- `no_safe_architecture`

That is an honest pre-final result for this block. It does not imply any final Stage 3 claim.

## Calibration

Because the selected-architecture artifact currently records `no_safe_architecture`, calibration is stopped honestly and the operating-point artifact records:

- `no_feasible_operating_point`

No fallback architecture or threshold was invented.

## Access control

The benchmark layer records phase-access audits.

Selection and calibration are blocked from final splits by the benchmark access policy.

Pre-final verification checks that no final evaluation artifacts already exist and that final splits were not accessed early.

## Runtime notes

Local commands completed in this block:

- `python examples/arcade_visual_video_discriminative_evidence_benchmark.py --verify-stage2-diagnosis`
- `python examples/arcade_visual_video_discriminative_evidence_benchmark.py --select-architecture`
- `python examples/arcade_visual_video_discriminative_evidence_benchmark.py --calibrate`
- `python examples/arcade_visual_video_discriminative_evidence_benchmark.py --verify-pre-final`

The expensive historical rebuild path remains distinct from lightweight verification and may still require a longer-running environment.
