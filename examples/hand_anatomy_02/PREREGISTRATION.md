# ZM-HAND-02 preregistration (frozen before the test split was scored)

Written after development on the **calibration split only** (`cal02`). The test split
(`test02`, 133 pairs + 50 pooled valid hands = 316 cases) had not been scored when this
file was written. Its SHA-256 is stored in `results.json`.

## Question

Can a VPM compiled from explicit anatomical topology and unexplained visual structure
distinguish an extra finger from a legitimate five-finger pose when global image
similarity cannot?

## Data (counterfactual pairs, independent blob generator, new seeds)

Each pair = the same hand (pose, proportions, rigid placement) rendered valid and with
exactly one defect. Categories: extra_finger (outer / between / adjacent / hidden),
duplicated_thumb, missing_finger, disconnected_wrist, abnormal_attachment.
`indistinguishable` = symmetric silhouette difference < 2% of the valid area (computed
from the two masks); those pairs are reported separately and never credited.
Valid pool for false-positive estimation = all valid twins + 10 kinds of unpaired valid
hands (includes close_overlap, proportions, partial_occlusion, sideways, curled).

## Methods (anomaly score; higher = more anomalous; computed from RAW VPM cells)

* **A_iou** (baseline): 1 - best library IoU (aligned, unrefined).
* **A_refined_iou** (baseline, controls for refinement): 1 - best IoU after constraint-respecting pose refinement.
* **C_residual** (primary): 1 - max over the 3 refined fits of `finger_residual_explainability`
  (1 - max finger-likeness of residual components; constants L_ref=28, E_ref=2.5, W_min=6,
  tolerance 2 px, attach dilation 3 px).
* Exploratory only (not part of the rule): B01_protrusion, C_residual_unrefined, C_branch.

## Rule (all conditions required to ACCEPT C_residual for extra-finger detection)

Primary set = test pairs with category extra_finger, subtype outer/between/adjacent,
not indistinguishable (48 pairs; the plan requires >= 40).

1. Thresholds: for every method, the smallest t with false-positive rate <= 0.10 on the
   **calibration** valid pool; applied unchanged to test. Detection = score > t.
2. Recall(C_residual) - Recall(baseline) >= 0.15 on the primary set, against **both**
   A_iou and A_refined_iou.
3. Exact two-sided sign test on discordant primary pairs, p < 0.05, against both baselines.
4. C_residual test false-positive rate on the valid pool <= 0.15.

Otherwise: **REJECT** for extra-finger detection. Other categories are reported
descriptively with the same thresholds; no multiple-comparison correction, no claims.
The test split is scored once; no constant, rule or method is changed afterwards.

## Development history on the calibration split (disclosed)

1. First version took the best-case residual over all library candidates with IoU >= 0.45
   (`margin` variants 0 to 0.2 were tried). Valid twins already left finger-like residuals of
   the same size as defects (library fit IoU ~0.79 even for valid hands), so C_residual
   scored worse than IoU (calibration primary recall 0.21-0.47 vs 0.53).
2. Added constraint-respecting pose refinement of the 3 best candidates and
   A_refined_iou as a second baseline. Calibration primary recall: A_iou 0.53,
   A_refined_iou 0.63, C_residual 0.32, C_residual_unrefined 0.42, C_branch 0.00 (in-sample thresholds).
   Paired separation on primary calibration pairs: A_iou 0.79, C_residual 0.58.
3. Tried residual of only the single best refined fit: no improvement (recall 0.42,
   paired 0.47). Not adopted. The design above was then frozen.

Expectation recorded before the test run: C_residual will most likely be rejected.
