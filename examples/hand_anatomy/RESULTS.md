# ZM-HAND-01 results

Everything below comes from `python -m examples.hand_anatomy.cli benchmark` (library
`sha256:ad0b25b9…`, 96 templates; calibration split 69 cases, test split 117 cases).
A second full run reproduced `results.json` exactly (apart from wall-clock time).
**All evidence is synthetic silhouettes. No real or AI-generated image was tested.**

## Headline

The hypothesis is **not supported at a useful level** by this implementation.

* The pipeline works end to end: template library -> alignment -> independent metrics ->
  genuine `ScoreTable`/`LayoutRecipe`/`VPMArtifact` -> declared decision contract ->
  report -> validate/replay. Identity, binding and tamper detection work.
* As a malformed-hand *detector* it is weak: on the independent test set with clear
  labels it flags 17 of 30 malformed hands (recall 0.57, 95% CI 0.39–0.73) at 1 false
  alarm in 50 valid hands. The plain best-IoU baseline flags 19 of 30 (0.63) at the same
  1 in 50. **Structured evidence did not beat the baseline there.**
* The specific idea motivating the experiment — an extra finger shows up as unexplained
  pixels even when a five-finger template overlaps most of the hand — **mostly failed
  for adjacent or overlapping extra fingers** (1 of 5 six-finger hands flagged). It
  worked for separated, large protrusions and for disconnected wrists.

## Test split, positive class = malformed (abstentions reported separately)

| Subset | Method | TP | FP | TN | FN | Abstained (mal/valid) | Precision | Recall (abstain = miss) | FPR (decided) | Coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| all (117) | A: best-IoU baseline | 22 | 1 | 61 | 33 | 0/0 | 0.957 | 0.400 [0.28, 0.53] | 0.016 | 1.00 |
| all | B: structured, calibrated | 23 | 1 | 61 | 30 | 2/0 | 0.958 | 0.418 [0.30, 0.55] | 0.016 | 0.98 |
| independent, clear labels (80) | A | 19 | 1 | 49 | 11 | 0/0 | 0.950 | 0.633 [0.46, 0.78] | 0.020 | 1.00 |
| independent, clear | B | 17 | 1 | 49 | 11 | 2/0 | 0.944 | 0.567 [0.39, 0.73] | 0.020 | 0.98 |
| independent, ambiguous defects only (10 malformed) | A / B | 1 / 0 | 0 | 0 | 9 / 10 | 0 | – | 0.10 / 0.00 | – | 1.00 |
| same-renderer, weak (27) | A | 2 | 0 | 12 | 13 | 0/0 | 1.00 | 0.133 [0.04, 0.38] | 0.00 | 1.00 |
| same-renderer, weak | B | 6 | 0 | 12 | 9 | 0/0 | 1.00 | 0.400 [0.20, 0.64] | 0.00 | 1.00 |

Confidence intervals are Wilson 95%; they overlap heavily, so A vs B is **not**
distinguishable except possibly on the small same-renderer subset (n = 15 malformed),
which is the weak test. Uncalibrated default thresholds (B0) flagged 35 of 48 decided
valid hands (FPR 0.73), so the thresholds in `DecisionContract` were meaningless until
calibrated. Ablations: disabling the finger-projection heuristic (B2) changed nothing
(calibration already turned it off); protrusion-only (B1) recalled 20 of 55.

Calibrated parameters (selected on the calibration split only, FPR cap 0.10):
IoU baseline threshold 0.74; contract `fit_floor 0.45, fit_ok 0.70, protrusion_floor
0.20, observed_floor 0.92, missing_floor 0.75, structure heuristic off`. On the
calibration split itself recall was 0.39 (A) and 0.55 (B), so B's selected rule has some
advantage in-sample that did not transfer to the held-out independent cases.

## Per defect class (test, independent generator, 5 cases each; flagged / n)

| Class | A: IoU baseline | B: structured |
|---|---|---|
| disconnected_wrist | 1 | **5** |
| large_protrusion | 3 | **4** |
| implausible_segments | **5** | 2 (+1 abstained) |
| missing_finger | **4** | 1 (+1 abstained) |
| duplicated_thumb | **3** | 2 |
| malformed_palm | **3** | 2 |
| six_fingers | 1 | 1 |
| abnormal_attachment | 0 | 0 |
| *valid hands falsely flagged* | 1 (spread) | 1 (spread) |

Same-renderer (weak): six_fingers A 2 / B 4; missing_finger A 0 / B 1; duplicated_thumb
A 0 / B 1.

## Which metrics discriminate? (single-metric AUC, test, independent, clear labels)

AUC = P(malformed scores lower than valid), best-ranked candidate per case. Descriptive
only; not used to tune anything.

| Metric | AUC |
|---|---|
| silhouette_agreement (IoU) | **0.89** |
| geometry_agreement | 0.86 |
| observed_explainability | 0.79 |
| protrusion_explainability | 0.71 |
| boundary_agreement | 0.71 |
| structure_agreement (projection count) | 0.68 |
| connectivity | 0.58 |
| missing_explainability | 0.53 |

IoU is the best single discriminator; the unexplained-structure metrics were weaker.
Malformed hands lower IoU anyway, and the library's imperfect coverage of valid hands
creates unexplained pixels in valid cases (below), so the two effects compete.
`connectivity` is near chance overall because it only helps for the one class
(disconnected wrist), where it is decisive (5/5).

## Demonstration cases (independent generator, not the template renderer)

| Case | Calibrated contract | Uncalibrated defaults |
|---|---|---|
| A: plausible open hand | `supported_plausible` (best IoU 0.805, explainer leaves protrusion_explainability 0.315) | **`suspected_anomaly`** (false alarm: the generator's thumb differs from every library thumb) |
| B: six-finger hand (extra outer finger, clearly visible) | `suspected_anomaly` (protrusion_explainability 0.00; extra finger = largest unexplained component) | `suspected_anomaly` |

Case B is detected, but only because the extra finger is separated from the others. Case
A shows why the result is fragile: a valid hand with a differently shaped thumb leaves
~3% of its area unexplained, and the calibrated protrusion floor (0.20) had to be lowered
to let such hands pass, which is exactly what lets adjacent extra fingers through. This
is a threshold-sensitive single-example demonstration, not evidence of performance.

## Answers to the research questions

**RQ1 (library recognises plausible silhouettes under declared transforms).** Partly.
For observations produced by the library's own renderer, alignment recovers
translation, scale (area-normalised) and in-plane rotation (IoU > 0.92, rotation error
<= 3 degrees; unit tests). For independently generated valid hands, best-IoU is only ~0.78
on average, and only a calibrated threshold avoids false alarms. Reference coverage,
not alignment, is the limiting factor.

**RQ2 (unexplained structure identifies malformed hands).** Only for separated or large
extra structure (disconnected wrist 5/5, large protrusion 4/5). The best-case explainer
absorbs adjacent or overlapping extra digits (six fingers 1/5, abnormal attachment 0/5,
duplicated thumb 2/5), and a missing finger is, by design, never called an anomaly.

**RQ3 (several measurements beat silhouette matching).** Not demonstrated. Differences
are within noise on the independent set (B 0.57 vs A 0.63 recall); a small advantage
appears only on the weak same-renderer subset. A thumb-variant library expansion
(3x thumb poses) gave no useful improvement in calibration-split discrimination
(protrusion AUC 0.65 -> 0.67), so more references of that kind were not adopted
(`build_library(thumb_variants=...)` remains as an option; test data were not used for
that decision).

**RQ4 (does ZeroModel give identity, traceability, replay, comparison).** Yes, and this
is the clearest positive result. Every decision is read from raw cells of a real
`VPMArtifact`; the report embeds the artifact and re-validates it; tampering with the
digest, a VPM value, or the decision is rejected (including a forged decision with a
recomputed digest); replay reproduces the artifact id and decision from original inputs
and flags a changed observation or library; recipes and templates change identities.
Attribution: decisions made from VPM cells equal decisions made directly from the
evidence in 186/186 cases, so **the VPM contributed zero detection gain**; all gains and
losses come from reference coverage, alignment, metrics and thresholds. Cost: ~10 KB of
JSON per VPM (162-byte core PNG).

**RQ5 (decision without an AI model at decision time).** Yes: rule evaluation over VPM
cells takes ~76 microseconds. The expensive part is evidence extraction (alignment +
comparison, ~0.6 s per case single-process, ~1.5 s when 4 workers share the CPU).

**RQ6 (what a single 2D silhouette cannot give).** Depth; self-occlusion order; whether
the palm or the back faces the camera; handedness (mirror-equivalent silhouettes); how
many digits are hidden by overlap; whether a missing region is occlusion or absence;
and whether a protrusion is a finger, a sleeve, or a segmentation artefact.

**RQ7 (what 3D templates would add).** Unmeasured here. They would resolve facing and
handedness and add pitch/yaw coverage, but the observed failures are about shape
coverage and overlap, not about the three yaw angles; this data does not show 3D to be
the next bottleneck.

**RQ8 (smallest next experiment).** Collect ~100 real generated-frame hand crops with
human masks and labels (normal, extra digit, missing digit, severe occlusion, unusual
plausible poses), hold out half before choosing thresholds, and score A vs B with
`evaluate-external`. If B still does not beat IoU, replace the fixed grid with a
continuous pose fit (optimise pose parameters per observation) before building 3D.

## Failure modes preserved (`benchmark/failures/*.png`, `results.json["failures"]`, 33 entries for B)

* False negatives (30): missing_finger 7 (counting same-renderer), duplicated_thumb 7,
  six_fingers 5, abnormal_attachment 5, malformed_palm 3, implausible_segments 2,
  large_protrusion 1. Most had a decent best-IoU (0.73–0.80) and "declared tests
  passed".
* Abstained malformed (2): best fit below the positive threshold.
* False positive (1): a spread-fingers valid hand left unexplained area (best-IoU 0.74).
* Hard-to-defend assumption: valid hands from the independent generator average ~0.78
  IoU against the nearest template, so "no explanation found" and "unusual valid pose"
  are not reliably separable with this library.

## What works / what is only synthetic / what is too strong

* Works: deterministic library, bounded alignment, VPM compilation, report/replay,
  CLI, overlays, abstention.
* Synthetic only: every accuracy number; the "independent" generator is still a
  hand-built silhouette synthesiser by the same author, so its independence is partial.
* Not performed: any real-image test; landmark-based comparison (implemented, unbenchmarked);
  Blender/MANO/3D (not implemented, no assets used).
* Assumptions too strong: that a 96-template grid covers valid variation; that a 2 px
  tolerance absorbs generator differences; that a calibration split of 69 cases yields
  stable thresholds (the selected floors sit at the lenient end of the grid).
