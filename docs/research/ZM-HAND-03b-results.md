# ZM-HAND-03b — Input quality and representation validation (results)

Exploratory development experiment, preregistered criteria in
[ZM-HAND-03b-preregistration.md](ZM-HAND-03b-preregistration.md). Development-side data only; the
held-out pool and the original 142 test crops were never opened. ZM-HAND-01/02/03a results are unchanged.
**Nothing here is independent held-out evidence.**

## Outcome

* **Formal conclusion (preregistered): DATASET UNSUITABLE for this process.** Good reference masks
  could not be obtained: 0 of 28 passed blind review, against a threshold of 20. No human reviewer was
  available, so no mask is human-verified.
* **Secondary, exploratory reading:** the representation is not hand-specific even on clean synthetic hands, so
  better masks alone would probably not rescue the silhouette approach. This is the part that supports the
  recommendation below, and it does not depend on mask quality.
* **Decision-table row:** "Good masks cannot be obtained". The row "verified masks still perform poorly"
  could not be tested, because verified masks do not exist.

## What was done

1. **Label-blind screening.** 168 development-side crops (three pools) viewed under anonymous ids with fixed
   criteria (single recognisable hand, at least 3 distinguishable fingers, at most about 30% occluded, blur not
   hiding the outline). 50 eligible, 118 excluded; every decision and reason is in
   `examples/hand_anatomy_03/data/screening_03b.json`.

   | Exclusion reason | Count |
   |---|---|
   | multiple or overlapping hands | 41 |
   | no hand / not a hand / sleeve only | 28 |
   | blur or too dark | 18 |
   | fingers not resolvable | 12 |
   | glove, illustration | 7 |
   | truncated, too small, other | 12 |

   Only 5 abnormal-labelled crops of at least 10 KB existed in the development pool, so pool 3 was drawn from the
   "unused" middle partition (still disjoint from the held-out pool) with a label-skewed quota, anonymised and
   shuffled before screening.
2. **Sample.** After the label reveal: 32 normal and 18 abnormal eligible, abnormal crops larger (median
   short side 334 vs 236 px). To neutralise size, 14 abnormal crops were chosen by hash order and each paired with
   the nearest-size normal crop: **28 images, 14/14, 28 distinct source videos**, median short side 294.5 vs
   276.5 px. Labels were used only for this balancing step.
3. **Reference masks.** Four blind model agents drew polygons (palm, fingers, up to the wrist; visible anatomy only,
   never "corrected"). Self-rated confidence: 7 high, 15 medium, 6 low.
4. **Blind review** by two other model reviewers using a strict standard. Result: **0 accept, 17 needs_fix,
   11 reject; 23 of 28 judged not finger-faithful, 5 unclear, none faithful.** Typical faults: merged curled
   fingers, loose polygonal edges, arbitrary wrist cuts, backgrounds or objects included, blurred hands
   with no recoverable fingers. The drawers' own "high" confidence did not predict acceptance.
5. **No fix round.** Even if all 17 needs_fix masks were repaired and accepted, 17 is below the preregistered 20, so
   the threshold is unreachable and a second round could not change the formal outcome.
6. **Human upgrade path.** `human_review_template_03b.json` and the 28 original/mask review images
   (`artifacts/zm_hand_03/mask_work/review/`) let a person accept, fix or reject each mask. Any such review would
   be the first human verification.

## Exploratory analysis (all 28 masks, not preregistered, masks NOT accepted)

Same pipeline as ZM-HAND-03a (forearm-free templates, 4 quadrant rotations, +/-45 degree alignment, 3-start
refinement, residual score). AUC = abnormal vs normal; 95% bootstrap intervals (2,000 resamples).

| Input | n eligible (abn/nor) | A_iou | A_refined_iou | C_residual |
|---|---|---|---|---|
| Automatic masks (u2netp) | 25 (12/13) | 0.49 [0.27, 0.72] | 0.55 [0.32, 0.77] | 0.48 [0.24, 0.72] |
| Reference masks (agent-drawn) | 28 (14/14) | 0.70 [0.49, 0.88] | 0.64 [0.41, 0.83] | 0.69 [0.49, 0.88] |
| Hull-filled reference masks (control) | 28 | 0.57 | 0.65 | 0.49 |

Simple baselines on the reference-mask set: image short side 0.45, mask solidity 0.49 (both chance, so the
size matching worked).

Agreement between automatic and reference masks is low (median IoU 0.63).

| Fit quality (median refined IoU) | Value |
|---|---|
| Real hands, reference masks | 0.823 |
| Real hands, automatic masks | 0.816 |
| Hull-filled reference masks (finger gaps removed) | 0.842 |
| Non-hand blobs (12 shapes) | 0.832 |
| **Clean synthetic valid hands (ZM-HAND-02, n = 261)** | **0.825** |

* **G1 (representation specificity) fails:** hand minus blob = -0.009, hand minus hull = -0.019 (required at
  least +0.05 and +0.03). Removing the finger gaps made the fit better, and a blob fits as well as a hand.
* **G2 (discrimination) not met:** best AUC 0.70 (A_iou) but its interval, [0.49, 0.88], includes 0.5, and
  it does not exceed the baselines by 0.10 with the required confidence.
* **Segmentation attribution:** the best AUC rises from 0.55 (automatic) to 0.70 (reference), a difference of +0.15.
  That is exactly at the declared "major bottleneck" threshold, but it is on partly different image sets, with wide
  intervals, using masks that failed review. Read it as "possibly some benefit from better masks", not a finding.

## Interpretation

* **Input quality is a hard limit here.** Automatic segmentation fails on skin-on-skin, blurred and cropped
  hands (median agreement 0.63 with the reference), and agent-drawn masks are not good enough to be a reference. Two
  imperfect mask sources disagree by almost as much as either differs from an ideal one. A finger-level
  silhouette study needs human-quality masks or a different input (resolvable, larger, unoccluded hands).
* **The representation, as measured by IoU, does not know hands from blobs.** This holds on clean synthetic
  silhouettes (0.825 vs 0.832), so the cause is not only bad masks. A 28-parameter refinement with four
  quadrant rotations can fit almost any compact shape to about 0.83. The earlier synthetic results (IoU
  helping in matched pairs but failing as an absolute threshold) are consistent with this: relative change
  within a hand is informative, absolute fit quality is not.
* **Why the hull control matters.** If the pipeline used finger structure, removing the finger gaps would lower
  the fit. It raised it. Finger structure is therefore not what drives the score.
* **Limits of this inference.** 28 images; blobs are only 12 simple shapes; reference masks are coarse and may
  themselves have merged fingers, which weakens the hull control specifically (but not the synthetic result).

## Recommendation

1. **Do not run a held-out evaluation of the silhouette inspector.** Neither the input side (masks) nor the
   representation (hand-specificity) supports it.
2. **Stop silhouette-template development** and test an explicit part-level representation next (finger and joint
   structure with relationships), starting on the ZM-HAND-02 synthetic pairs: it must separate hands from
   blobs and valid hands from their defective twins *before* any real image is touched. Keep the blob control, the
   hull control and the size/solidity baselines as mandatory comparisons.
3. **If real images are used next, get human-quality input first:** a person reviews or corrects masks using the
   existing review images and template, or use hands that are resolvable (large, unoccluded, with wrist). The
   screening shows such hands are uncommon even among crops preselected for size: 50 of 168 (30%) qualified.
4. Reporting rule for the future: a reference mask produced by a model must be labelled model-made unless a human
   has reviewed it.

## Limitations and disclosures

* Masks were drawn and reviewed by models. The reviewers were instructed to be strict and may be stricter than a
  human; a human review could accept some of the 17 needs_fix masks. It could not realistically bring the count
  to 20 without redrawing.
* Selection used labels only for the size-matched balancing step; screening was label-blind. The sample was then
  used to decide next steps, so it must not be reused as test data.
* The earlier ZM-HAND-03a `drop_forearm` option remains default-off; no frozen result changed.

## Reproduce

```powershell
python -m examples.hand_anatomy_03.exp03b --exploratory-all     # needs the artifacts listed below
python -m pytest examples/hand_anatomy_03/tests -q
```

Inputs (git-ignored `artifacts/zm_hand_03/`): `vbench_pool*/` crops (about 5.5 MB fetched by range request), `mask_work/`
(images, polygons, masks, review composites), `seg/u2netp.onnx`, the isolated `venv_seg`. Committed:
`examples/hand_anatomy_03/data/{screening_03b,sample_03b,sample_03b_labels,model_review_03b,human_review_template_03b}.json`,
`mask_tool.py`, `exp03b.py`. Figures: `artifacts/zm_hand_03/exp03b/paired_overlays.png` (original, automatic mask,
reference mask, fit).
