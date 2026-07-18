# Video Stage Three Diagnosis

Date: July 18, 2026

Status: post-hoc diagnosis of frozen Stage 2 evidence only

Stage 2 parent commit: `d00e18b67fbe2f62617cd0ac47c7ee2f63487cb8`

Frozen Stage 2 benchmark digest: `sha256:589bb074e1b53b06657cfb75bf7b8d67eae43cc5f76e7237ab07f23ccca49c75`

Frozen Stage 2 split digest: `sha256:d25b694b3cce93bf93f58239163331f3f6370d32a2b5cce53b4541902b0f8c23`

## Scope

This document diagnoses the frozen Stage 2 local-correlation failure mode without changing or superseding the Stage 2 result. The diagnosis uses the Stage 2 provider, calibration, benchmark fixture, and evaluation logic exactly as frozen.

Machine-readable outputs for this diagnosis are committed under:

`docs/results/video-discriminative-local-evidence-v1/diagnostics/`

## Core findings

### 1. Exact benign frames failed because visibility semantics were wrong for the chosen region contract

For `video-final-benign-exact`, the top-1 row was usually the correct row and the top-1 distance was `0.0`, but the decision was still rejected for `minimum_visible_fraction`.

The dominant cause was the Stage 2 `cooldown_indicator` region:

- mean visible fraction: `0.5`
- min visible fraction: `0.5`
- mean distance: `0.0`

The frozen Stage 2 operating point required `minimum_visible_fraction = 0.8`, so the exact row was rejected even when the region matched perfectly on its informative pixels. This is not semantic uncertainty. It is a geometry-driven veto caused by a very small region whose registration overlap cannot exceed `0.5` under the chosen translation regime.

### 2. Translated benign frames failed for the same reason before match evidence became the bottleneck

For `bounded_translation_photometric`, all 22 frames were rejected by `minimum_visible_fraction`.

The same `cooldown_indicator` region again had:

- mean visible fraction: `0.5`
- mean distance: `0.0`

The provider therefore discarded exact current-frame evidence before distance or action-separation thresholds had a chance to help.

### 3. Occluded benign frames exposed both visibility and evidence-collapse problems

For `bounded_translation_occlusion`:

- `winner_threshold` caused 20 rejections
- `minimum_visible_fraction` caused 2 rejections

This family showed that once visibility no longer vetoed everything, the weighted-mean distance aggregation could still fail. The `target_band` distance rose materially while `tank_band` and `cooldown_indicator` often stayed near zero, so a single scalar aggregate erased the distinction between strong support, localized contradiction, and partial evidence availability.

### 4. One poor region could veto otherwise sufficient evidence

Stage 2 defined candidate visibility as the minimum visible fraction across all regions. In practice this let the weakest region veto the whole frame. The exact and translated benign families show the failure cleanly:

- `target_band` stayed near `0.964`
- `tank_band` stayed near `0.964`
- `cooldown_indicator` stayed at `0.5`
- the candidate visible fraction became `0.5`
- the frame was rejected despite exact top-1 row recovery

### 5. Correct-row rank was often good even when exact acceptance was impossible

Across the Stage 2 final video benchmark:

- correct row in top-1: `55 / 132`
- correct row in top-2: `62 / 132`
- correct row in top-3: `64 / 132`
- correct row in top-5: `64 / 132`

For action:

- correct action in top-1: `64 / 132`
- correct action in top-5: `64 / 132`

This means the Stage 2 reader often preserved meaningful current-frame structure, but the top-one exact-acceptance contract plus minimum-visibility veto converted that structure into zero utility.

### 6. Stage 2 final rejection reasons were dominated by calibration gates, not absence of current-frame similarity

The frozen Stage 2 rejection profile remained:

- `minimum_visible_fraction`: `112`
- `winner_threshold`: `20`

The diagnosis sharpens that summary:

- exact and translated benign failures were mostly not due to poor row ranking
- several failures happened with exact top-1 row and zero top-1 distance
- the bottleneck was visibility semantics and scalar collapse, not only raw similarity

### 7. Background and non-discriminative pixels likely diluted evidence

Stage 2 distances were computed over full rectangular region crops after registration. The diagnosis does not yet prove how much of each rectangle is truly discriminative, but it shows two strong symptoms:

- exact and translated cases can have zero distance in the best row while still failing because a tiny region constrains visibility globally
- occluded benign cases can accumulate enough regional distortion in one area to trip `winner_threshold` despite good support elsewhere

That pattern is consistent with non-discriminative pixels being allowed to dominate regional distance and weighted averaging blurring support and contradiction.

## Required Stage 2 questions answered

1. Why did exact benign frames fail V2?
   The exact row was usually top-1 with distance `0.0`, but the tiny `cooldown_indicator` region forced candidate visibility to `0.5`, below the frozen `0.8` threshold.

2. Why did translated benign frames fail?
   The same region-level visibility rule rejected all translated-photometric frames even when regional distances were still `0.0`.

3. Why did occluded benign frames fail?
   A minority failed visibility directly, but most failed `winner_threshold`, indicating that weighted scalar aggregation could not preserve localized support under partial corruption.

4. Which regions caused `minimum_visible_fraction` rejection?
   Primarily `cooldown_indicator`, with some contribution from `target_band` in the occlusion family.

5. Was visible fraction reduced by genuine occlusion or registration overlap?
   Often by registration overlap alone. Exact and translated benign frames still produced `0.5` visible fraction in `cooldown_indicator` with zero distance.

6. How much of each region was actually discriminative between rows?
   Stage 2 did not represent this explicitly. That missing contract is now a Stage 3 design requirement.

7. How much of each region distinguished different actions?
   Stage 2 did not represent this explicitly either. Action-conflict evidence must be separated in Stage 3.

8. Did background pixels dominate any regional distance?
   The diagnosis supports that hypothesis, especially in occluded cases, but Stage 2 lacked a discriminative-mass model, so this remains a Stage 3 falsifiable question rather than a concluded fact.

9. Did the correct row commonly appear in the top two, top three, or top five candidates?
   Yes. Top-2 recall was `62 / 132`; top-3 and top-5 recall were both `64 / 132`.

10. Did the correct row commonly remain inside a small candidate set despite top-one rejection?
    Yes, often within top-2 or top-3. Stage 2 simply had no safe candidate-set output contract.

11. Were conflicting-action candidates closer than same-action alternatives?
    Often no. Same-action competitors dominated many top-5 lists. That means some ambiguity is row-local rather than action-conflict-driven.

12. Did weighted averaging hide individual strongly supportive regions?
    Yes in the sense that Stage 2 flattened all region evidence into one scalar distance and one minimum visibility, losing the distinction between support, contradiction, and informative mass.

13. Did one poor region veto otherwise sufficient evidence?
    Yes. The `cooldown_indicator` region is the clearest example.

14. Would corrected visibility semantics alone have changed feasibility?
    Likely yes for exact and translated benign utility. It would not by itself resolve all occlusion and ambiguity failures, so it is necessary but probably not sufficient.

15. Would candidate-set preservation have exposed useful temporal disambiguation opportunities?
    Yes. The top-k rank results show that Stage 2 frequently retained the correct row near the top even when exact acceptance failed.

## Interim conclusion for Stage 3

Visibility semantics were a real Stage 2 bottleneck, but not the only one.

Stage 3 should test at least four separable hypotheses:

- corrected visibility over informative aligned evidence
- discriminative rather than rectangular evidence mass
- separate support and contradiction accounting
- candidate-set preservation before temporal narrowing

Stage 3 must still prove those changes materially help on a fresh final split. The diagnosis only justifies the next architecture search.
