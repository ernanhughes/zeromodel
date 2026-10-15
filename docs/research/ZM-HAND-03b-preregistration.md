# ZM-HAND-03b preregistration — input quality vs representation (exploratory, development-side)

Frozen before any reference mask was run through the comparison pipeline. This is an
**exploratory development experiment, not a held-out claim**: the sample is used to decide what to
do next, so it must never be quoted as independent evidence.

## Question

Is the failure of ZM-HAND-03a caused by the visual evidence entering ZeroModel (segmentation, image
quality) or by the anatomical representation inside it?

## Data and selection (all development-side; nothing from the held-out pool)

* Pools drawn from the video-disjoint split of ZM-HAND-03a: development pool (pool 1, pool 2) and the
  "unused" middle partition (pool 3). The held-out pool (last 50% of videos by hash) was not opened.
* Visual eligibility screened **label-blind** (anonymous ids and image sizes only), criteria fixed before
  judging: single recognisable hand as main subject; at least 3 fingers individually distinguishable;
  at most about 30% occluded; blur not hiding the outline; wrist preferred, not required. All 168
  screening decisions with reasons are in `examples/hand_anatomy_03/data/screening_03b.json`
  (50 eligible).
* Labels were opened only after screening. 14 abnormal crops by hash order, each paired with the
  nearest-size unused normal crop (size matching neutralises the image-size confound; labels used only for
  this balancing). Result: 28 images, 14/14, 28 distinct source videos.
* Pool 3 was drawn with a label-skewed quota (40 abnormal / 20 normal) because large abnormal crops are
  rare in the development pool; it was anonymised and shuffled before screening.

## Masks

* **A (automatic):** `u2netp` exactly as in ZM-HAND-03a (checkpoint sha256 `309c8469...`), unchanged.
* **V (reference):** drawn as polygons by model agents blind to labels, then checked by a separate blind
  model reviewer. **No human reviewer was available, so V is agent-drawn and model-reviewed. It is NOT
  human-verified and must not be described as such.** A human-review sheet and a response template are
  provided so the claim can be upgraded later.
* Only reviewer-accepted V masks enter the V analysis; rejections are listed, not silently dropped.

## Methods (fixed; no tuning)

Forearm-free templates, four quadrant rotations, the +/-45 degree alignment, three-start refinement, and
the ZM-HAND-02 residual score, i.e. `real_pipeline.assess_one` exactly as run in ZM-HAND-03a. Scores
(higher = more anomalous): `A_iou`, `A_refined_iou`, `C_residual`. Same settings for A and V masks.

## Controls

1. 12 non-hand blobs (ellipse, rounded rectangle, L, blob clusters) already generated in ZM-HAND-03a.
2. **Convex-hull control:** each accepted V mask filled to its convex hull (finger gaps removed).
3. Simple baselines on V masks: image short-side (size) and mask solidity (area / convex-hull area).

## Decision criteria (declared now)

* **G1, representation specificity (V masks):** median refined IoU of hand masks exceeds the blob
  median by at least 0.05 **and** exceeds the median of the hull controls by at least 0.03.
* **G2, discrimination (V masks):** the best AUC among {`A_iou`, `A_refined_iou`, `C_residual`} is at
  least 0.70, its bootstrap 95% lower bound (2,000 resamples, seed 7) is above 0.50, and it exceeds both
  the size and solidity baselines by at least 0.10.
* **Segmentation attribution:** best-method AUC on V masks minus best-method AUC on A masks, same
  images. A difference of at least 0.15 is read as "segmentation is a major bottleneck"; smaller is
  "not clearly".
* **Dataset suitability:** if fewer than 20 reviewer-accepted V masks are obtained, the available data is
  declared unsuitable.

Outcome mapping:

| Result | Conclusion |
|---|---|
| Fewer than 20 accepted V masks | Dataset unsuitable |
| G1 and G2 both met | Proceed to a preregistered held-out evaluation |
| G1 or G2 not met on V masks | Stop silhouette-based development; recommend a landmark/part-level experiment |
| V improves over A by at least 0.15 | Report segmentation as a bottleneck, in addition to the above |

With 28 images a 95% interval on an AUC is about +/-0.2, so even a "met" outcome only justifies a
properly powered, preregistered held-out test. A failure is informative about this pipeline on this data,
not about every possible hand representation.
