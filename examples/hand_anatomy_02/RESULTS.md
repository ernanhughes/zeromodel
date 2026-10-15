# ZM-HAND-02 results

All data are synthetic counterfactual pairs. Test split scored once under the rule in
[PREREGISTRATION.md](PREREGISTRATION.md) (its hash is stored in `results.json`).
ZM-HAND-01 sources and results are unchanged.

## Verdict

**REJECT.** The residual/topology approach (`C_residual`) did not improve extra-finger
detection over plain silhouette IoU; it was clearly worse.

Primary set = 48 distinguishable extra-finger test pairs (outer 15, between 17, adjacent 16).
Thresholds were set on the calibration valid pool at 10% false positives and applied unchanged.

| Method | Detected / 48 | Recall [95% CI] | Test FPR (183 valid) | Gain vs A_iou (sign test p) | Gain vs A_refined_iou (p) |
|---|---|---|---|---|---|
| A_iou (baseline) | 25 | 0.52 [0.38, 0.66] | 0.169 | – | – |
| A_refined_iou (baseline) | 29 | 0.60 [0.46, 0.73] | 0.087 | – | – |
| C_residual (primary) | 15 | 0.31 [0.20, 0.45] | 0.126 | -0.21 (p = 0.087) | -0.29 (p = 0.004) |
| C_residual_unrefined | 19 | 0.40 | 0.131 | -0.12 (p = 0.31) | -0.21 (p = 0.021) |
| B01_protrusion (ZM-HAND-01 metric) | 15 | 0.31 | 0.126 | -0.21 | -0.29 |
| C_branch (skeleton branch excess) | 0 | 0.00 | 0.000 | -0.52 | -0.60 |

Rule conditions failed: recall gain >= +0.15 (observed -0.21 / -0.29) and p < 0.05 in the
favourable direction. Caveat on the comparison: false-positive rates are matched on the
calibration split, not on test. A_iou's test FPR (0.169) exceeds the 0.10 cap, which flatters
its recall; A_refined_iou, with the lowest test FPR (0.087), still detected more than
C_residual (0.126). The conclusion does not depend on that mismatch.

Unpaired AUC, distinguishable defects vs valid pool: A_refined_iou 0.85, A_iou 0.79,
C_residual_unrefined 0.64, B01 0.57, C_residual 0.57, C_branch 0.48.

## Per group (test; detected / pairs, paired separation = P[defect scores above its own valid twin])

| Group | A_iou | A_refined_iou | C_residual |
|---|---|---|---|
| extra_finger / outer | 9/15 (1.00) | 13/15 (1.00) | 10/15 (0.83) |
| extra_finger / between | 9/17 (0.82) | 9/17 (0.65) | 3/17 (0.26) |
| extra_finger / adjacent | 7/16 (0.88) | 7/16 (0.94) | 2/16 (0.44) |
| extra_finger / hidden [indistinguishable] | 1/10 (0.50) | 0/10 (0.50) | 0/10 (0.50) |
| extra_finger / between, adjacent [indistinguishable] | 0/27 | 0/27 | 2/27 |
| duplicated_thumb | 5/12 (0.92) | 5/12 (0.92) | 4/12 (0.42) |
| missing_finger | 10/12 (1.00) | 8/12 (1.00) | 3/12 (0.38) |
| disconnected_wrist | 8/12 (1.00) | 11/12 (1.00) | 2/12 (0.54) |
| abnormal_attachment | 3/7 (1.00) | 2/7 (1.00) | 0/7 (0.14) |

Counts reconcile: 133 test pairs (48 primary + 85 other categories / indistinguishable),
266 pair members + 50 pooled valid hands = 316 cases; valid pool for FPR = 133 twins + 50 = 183.
Indistinguishable pairs are never credited; any detection there is spurious.

## What this establishes

1. **A residual-and-topology VPM was built and is fully traceable** (raw cells, bindings, replay,
   tamper rejection, forged-decision rejection are tested), but **did not carry useful
   detection information**. As in ZM-HAND-01, traceability is the contribution, not accuracy.
2. **Global similarity did respond to the extra finger.** Within a pair, IoU scored the defect
   as more anomalous than its own valid twin in 82–100% of distinguishable extra-finger pairs
   (separation column). The earlier premise that a good overall match swamps a sixth finger was
   not what limited detection here. What limited it was **between-hand variation**: valid hands
   vary by roughly 0.08 in IoU (0.73–0.81 in calibration), the same size as the drop caused by
   one added finger (~0.04), so any fixed threshold on one image misses about half.
3. **The residual score was at or below chance within pairs** for between/adjacent (0.26 /
   0.44) and weaker than IoU for outer (0.83 vs 1.00). Hypotheses (not tested separately): the joint-limited
   refinement can still absorb an extra finger by recruiting a neighbour's abduction and
   thickness (visible in `pair_sheet.png`); and the top residual is often a thin attached
   sliver along an edge, produced by generator-versus-capsule shape differences, which the
   finger-likeness product cannot tell from a digit.
4. **Skeleton branch counting added nothing** (0/48; its calibrated threshold was the maximum
   possible), consistent with merging of overlapping fingers.
5. **More capacity did not mean better detection.** Refinement raised both valid and defective
   IoU about equally and improved the IoU baseline (0.52 -> 0.60), while residual analysis on the
   same fits got worse (0.40 unrefined -> 0.31 refined). This supports the concern that extra
   flexibility can let malformed hands be explained away.

## Answer to the sharp question

No: in this implementation and data, a VPM built from explicit topology and residual evidence
did **not** distinguish an extra finger from a legitimate pose better than global similarity.
The ZM-HAND-01 negative finding stands and is strengthened.

## Development disclosures

* Methods were developed and tuned on the calibration split only (history in the
  preregistration). The test split was scored once; the final per-case records were saved.
* A key-name bug in the analysis code (`valid` vs `valid_twin`) surfaced after test scoring.
  It was fixed and the evaluation re-run from the saved per-case records; no method, constant
  or threshold changed. `benchmark --reuse-records` reproduces `results.json` from them.
* The refinement stage was added after calibration diagnostics showed valid twins left
  finger-like residuals comparable to defects. It was not in the original brief and is
  reported with its own baseline (A_refined_iou).

## What would be worth doing next (hypotheses, not results)

* **Reference-based inspection.** The one setting where IoU separation was near-perfect is
  the paired one: the same hand, one change. Film frames provide this naturally (the same
  hand in neighbouring frames). A temporal-consistency experiment on real generated clips
  is the most promising direction suggested by this data.
* **Real generated crops remain mandatory** before any production claim. Nothing here
  addresses real-image segmentation error.
* Residual analysis would need a fit that cannot absorb extra digits (tighter, per-digit
  anatomical priors rather than broad ranges) and a residual classifier that separates edge
  slivers from digits; both are unproven.

## Terminology correction

The code, `results.json` and the frozen preregistration use the label `indistinguishable`
for pairs whose silhouettes differ by less than 2% of the valid area. That word is wrong: a
small extra digit can be obvious to a human and topologically important while changing the
area very little. Read it as **low-silhouette-difference**. Identifiers were left unchanged so
the scored artifacts and the preregistration hash stay exactly as run. Those pairs are a
separate evaluation group that is reported, not discarded, but the 48-pair primary set that
excludes them is a diagnostic subset, not the full distribution of malformed hands. The
no-credit rule applied to them only because the preregistration said so; it is not a claim
that those defects are undetectable in general.
