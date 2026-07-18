# Video Stage Three Evidence Mechanics

Date: July 18, 2026

Status: implementation note for frozen Stage 3 evidence mechanics

Parent documents:

- [video-stage-three-preregistration.md](/C:/Projects/zeromodel/docs/research/video-stage-three-preregistration.md)
- [video-stage-three-operational-contract.md](/C:/Projects/zeromodel/docs/research/video-stage-three-operational-contract.md)
- [video-stage-three-diagnosis.md](/C:/Projects/zeromodel/docs/research/video-stage-three-diagnosis.md)

## Scope

This note records implementation details for the first Stage 3 mechanics block.

It does not change the frozen preregistration or the operational contract.

## Stage 2 audit summary

The frozen Stage 2 registration implementation in [visual_registration.py](/C:/Projects/zeromodel/zeromodel/visual_registration.py) enumerates all integer translations inside `[-max_dx, max_dx] x [-max_dy, max_dy]`, computes centered normalized overlap distance on the valid aligned rectangle, rejects translations below the configured overlap floor, and selects by:

1. lower distance
2. smaller Manhattan displacement
3. smaller absolute vertical shift
4. smaller absolute horizontal shift
5. deterministic signed ordering through `(dy, dx)`
6. overlap fraction only after those earlier terms

The frozen Stage 2 provider in [video_local_correlation.py](/C:/Projects/zeromodel/zeromodel/video_local_correlation.py) uses the returned overlap fraction directly as region visibility and then takes the minimum across regions, which is why a tiny exact-match region could still report visible fraction `0.5` and veto the frame.

Stage 3 therefore adds a separate versioned registration layer instead of changing Stage 2 semantics.

## Frozen Stage 3 registration layer

`register_informative_translation(...)` keeps Stage 2 distance as the primary objective and freezes:

- tie epsilon: `1e-12`
- signed offset ordering: ascending `(dy, dx)`

Among equal-distance translations, the deterministic key is:

1. greater available informative mass
2. greater valid pixel count
3. greater geometric overlap
4. smaller Manhattan displacement
5. smaller absolute vertical shift
6. smaller absolute horizontal shift
7. ascending `(dy, dx)`

The registration result preserves both the selected translation and enough runner-up data to explain why the winner defeated the next-best equal-distance alternative.

## Mask materialization

`build_discriminative_masks(...)` creates one immutable mask payload per prototype row.

For each row:

- `row_informative_weights` marks pixels that differ from at least one competing row by more than the frozen intensity tolerance.
- `action_conflict_weights` marks pixels that differ from at least one conflicting-action row by more than the same tolerance.
- `stable_weights` is derived only from permitted development observations for that row.
- when no development observations exist for a row, stability falls back to zero mass rather than assuming full stability.
- `separation_weights` uses the nearest relevant prototype separation, clipped to `[0, 1]` by `separation_cap`.

The payload digest includes the spec digest plus each array's shape, dtype, and contiguous bytes.

## Region evidence extraction

`extract_candidate_region_evidence(...)` uses the candidate's informative-and-stable mass for registration, then separates:

- support: candidate pixel strictly closer than every competing row
- contradiction: some competing row strictly closer than the candidate
- critical contradiction: some conflicting-action row strictly closer than the candidate on conflicting-action mass
- conflicting-action support: candidate strictly closer than every conflicting-action row

All per-pixel comparisons are deterministic and use the registered aligned slices only. No hidden labels, temporal truth, or final-split annotations participate in the evidence computation.
