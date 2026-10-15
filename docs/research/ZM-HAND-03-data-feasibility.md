# ZM-HAND-03 — Real-data feasibility assessment

Status: feasibility only. No temporal VPM, tracker, or new package was built. ZM-HAND-01 and
ZM-HAND-02 sources and results are unchanged. All counts below come from files fetched on
2026-10-14 and the scripts in `examples/hand_anatomy_03/`; keyword-derived classes **nominate
candidates for manual review and verify nothing**.

## Verdict

* **Spotlight cannot power the temporal / hand-structure study as specified.** It has 600
  videos and 1,604 error intervals, but only 39 anatomy intervals mention a hand/finger/wrist
  term, 15 name a digit part, and about 8 make an explicit finger-count claim. Held-out
  (prompt-level split) holds 14 anatomy-hand intervals in 13 clips, and 2 explicit finger-count
  claims. Even the broadest nomination (128 intervals, 97 clips) is unverified and mostly not
  hand-structure defects. It has no hand masks and no verified-normal intervals.
* **VBench-2.0 hand data can support a small real-image test of the existing single-frame
  inspector**, reachable for about 3.2 MB (not 10.9 GB). It has 142 labelled test crops (72
  abnormal / 70 normal), no pixel masks, no timestamps. **Blocker: a hand mask provider.**
* **Recommendation:** do not build temporal code yet. Run a preregistered single-frame
  real-image test on the VBench crops first (section 8), after resolving the mask blocker.

## 1. Accessible datasets and what was actually fetched

| Item | Source | Size | Purpose |
|---|---|---|---|
| Spotlight `test/metadata.jsonl` | `UBC-ViL/Spotlight-VideoGen-Errors` | 1.17 MB | annotation inventory (**the only authorized download**) |
| Spotlight dataset card, file listing with sizes | HF API / README | ~0.04 MB | licence, structure, per-video sizes |
| VBench-2.0 human-anomaly card, file listing with sizes | HF API / README | ~0.01 MB | structure, sizes |
| VBench ViTDetector README (data format, licence text) | GitHub raw | 9 KB | label format |
| **Tail of `opensource.zip`, HTTP range read** | HF resolve URL | **18.0 MB** | zip central directory (file inventory) only |
| `hand_test.jsonl` (range read of 2.8 KB, 2,178 B compressed) | `opensource.z01` | 2.8 KB | test labels |

Disclosure: the 18 MB central-directory read and the 2.8 KB label read go beyond the single
named file. They are metadata-only range reads (0.16% of the 10.9 GB set); no video and no image was
fetched, and nothing was extracted except `hand_test.jsonl`. If that was outside what you
intended, the raw files are only under git-ignored `artifacts/zm_hand_03/raw/` and can be deleted.

Not downloaded: any Spotlight video (3.98 GB total, mean 6.6 MB), any VBench image or video,
`hand_train.jsonl`.

## 2. Spotlight annotation counts (all from `metadata.jsonl`)

600 videos, 200 unique prompt ids (`pid`), 200 videos per generator (Veo 3, Seedance, LTX-2),
1,604 intervals. Only a `test` split exists, so every partition is ours to define.

| Type | Intervals |
|---|---|
| adherence | 455 |
| physical_viol | 287 |
| app_disapp | 278 |
| motion | 226 |
| anatomy_body_pose | 183 |
| logical | 175 |

By generator: LTX-2 580, Seedance 517, Veo 3 507 intervals. Difficulty: easy 793, medium 396,
hard 415. Interval duration: median 1.9 s (p25 1.0, p75 3.5, max 8.0); 326 under 1 s, 772 of 1-3 s,
506 of 3 s or more. Video length is not in the metadata, so "persistent versus transient" can only
be proxied by duration until videos are read.

### Hand-related nominations (keyword match with word boundaries; unverified)

| Class | Intervals | Clips | Notes |
|---|---|---|---|
| `candidate_hand_anatomy` (anatomy_body_pose + hand/finger/wrist/palm term) | 39 | 35 | of the 183 anatomy intervals |
| of which name a digit part (finger, thumb, digit, knuckle) | 15 | | |
| of which digit part + structure term (extra, missing, fused, ...) | 9 | | |
| `candidate_hand_appearance` (app_disapp + hand term) | 37 | | 20 of the 128 candidates last 3 s or more |
| `candidate_hand_motion_or_contact` (motion/physical + hand term) | 52 | | mostly contact/physics, not static anatomy |
| `requires_manual_review` (arm-only terms, or hand terms under adherence/logical) | 63 | | includes 12 arm-only anatomy intervals |
| **Candidate total (the three classes above)** | **128** | **97** | 68 prompt ids; LTX-2 43, Seedance 49, Veo 3 36 |

Reading the 15 digit-part anatomy annotations by hand (judgement aid only, not verification):
about 8 are explicit finger-count claims ("too many fingers", "missing a finger", "3 instead of
4", fingers that flicker between 3 and 5). The rest describe length, rigidity, uniform girth,
blur, or a skin flap. The other 24 hand-term anatomy annotations are shape/warp/morph complaints
(hand elongates, warps, bends impossibly). So explicit extra/missing-digit defects, the case the
ZM-HAND experiments targeted, are on the order of **8 intervals in 600 videos**.

Caveats on the counts: the keyword list can miss hands described without those words and can
match non-defect mentions; annotators flagged salient errors only, so unannotated hand regions
may also be defective. Scenes are often multi-person (about 10% of prompts name groups), so hands
can be small; there is no hand detector or mask in the dataset.

## 3. Negatives

* **No verified-normal hand intervals exist.** An absent annotation does not mean "anatomically
  correct". Annotators did not mark normal hands, and unmarked defects are possible.
* Only 21 prompts mention hands; 29 videos with hand-mentioning prompts carry no hand-term
  annotation. These are *candidates for negative review*, not negatives.
* Required before any false-positive claim: human verification that a hand is visible and
  normal for a stated interval, including unusual angles and partial occlusion, recorded in a
  separate negative-annotation file. This must be done by a person before the held-out score
  is computed, on held-out prompts only.

## 4. Licensing and storage

| Dataset | Licence statement found | Caveat |
|---|---|---|
| Spotlight | CC BY-NC 4.0 (HF card and tags) | Non-commercial: research use under attribution is fine; do not assume it permits commercial filmmaking use. Generator terms (Veo 3, Seedance, LTX-2) on the videos were not checked. |
| VBench-2.0 human anomaly | HF card has **no licence field**. The VBench ViTDetector README states MIT, free for academic and commercial use with attribution. The code repository's LICENSE file is Apache-2.0. | Three different statements. The archive also contains 1,020 real-world video files (`vad_real`) and CogVideoX outputs whose upstream rights are not stated. Verify before any redistribution or commercial use. This is not legal advice. |

Storage: Spotlight full 3.98 GB; the 97 broad-candidate clips 562 MB; the 35 strict hand-anatomy
clips 199 MB; all 141 clips with any hand/arm term 835 MB (lists in `examples/hand_anatomy_03/data/`).
VBench full 10.9 GB (3 zip segments); archive holds 126,011 JPEGs (3.8 GB compressed), of which
18,210 are hand crops (about 199 MB), plus 2,286 video files (about 7 GB). The 142 test crops
total 3.2 MB and live in the first segment, `opensource.z01`.

## 5. VBench-2.0 hand data

Structure (from the official README): `dataset/hand_train.jsonl`, `hand_test.jsonl`, `all_images`
(individual JPEGs), plus source videos from CogVideoX-1.5, CogVideoX-1.0 and real-world videos.
Each label row is `[image_filename, label, score]`; the README describes 0 as normal and 1 as
abnormal; `score` is unused. Crops come from a YOLO-World detector (hand boxes made square).

Verified from `hand_test.jsonl` and the central directory: 142 rows, all 142 images present;
72 abnormal, 70 normal; 77 distinct source-video ids (up to 6 crops per video; 13 videos have
both labels). Not verified: `hand_train.jsonl` contents, whether train/test are split by video
or by frame (the filename pattern `<video-hash>_<frame>_hand_<k>.jpg` suggests a leakage
check is needed), the exact definition of "abnormal", and which test crops are from real versus
generated videos.

Not provided: pixel masks, temporal intervals, handedness, or forearm context. Crops are
tight square boxes, but ZM-HAND-01/02 templates include a forearm stub, so a crop without
wrist/forearm will mismatch the template family independent of any anatomy question.

## 6. Proposed split (Spotlight, if it is ever used)

Prompt-level, deterministic: SHA-256 of `"zm-hand-03:" + pid`, rank all 200 pids, first 40%
development (80 pids), next 20% calibration (40), last 40% held-out (80). Videos from the same
prompt, across all three generators, always share a partition. Saved as
`examples/hand_anatomy_03/data/spotlight_pid_split.json`. Candidate nominations per partition:

| Class | Development | Calibration | Held-out |
|---|---|---|---|
| candidate_hand_anatomy | 16 intervals / 14 clips | 9 / 8 | **14 / 13** |
| candidate_hand_appearance | 17 / 15 | 7 / 6 | 13 / 10 |
| candidate_hand_motion_or_contact | 20 / 19 | 8 / 6 | 24 / 21 |
| requires_manual_review | 23 / 21 | 11 / 11 | 29 / 23 |
| All 128 candidates | 53 / 41 | 24 / 18 | **51 / 38** |

Explicit finger-count claims by partition: held-out 2 (`sid_011_ltx2`, `sid_162_ltx2`),
calibration 1, development 5. Held-out labels and clips must stay out of development and
threshold selection. The hash split was fixed before inspecting any annotation text, but while
characterising digit-part cases I read the annotation text of two held-out intervals
(`sid_011_ltx2`, `sid_162_ltx2`). They should be re-reviewed blind, or dropped from held-out
scoring, if held-out labels are ever used.

For VBench the proposed split is by source-video id, not by image: calibrate on a video-level
subset of train crops, test on the 142 test crops after confirming no video id overlaps train.

## 7. Is the experiment sufficiently powered?

Provisional requirements: at least 60 labelled defect intervals, at least 15 held-out clips,
persistent and transient reported separately, at most 10% false positives on held-out verified
negatives, and a recall gain of at least 10 points over refined single-frame IoU.

| Question | Assessment |
|---|---|
| Hand-structure defects (extra/missing/fused digits) from Spotlight | **Underpowered.** About 8 explicit finger-count intervals in total, 2 held-out. 39 hand-anatomy nominations, 14 held-out in 13 clips (below 15), none verified. |
| Broad hand nominations (128 intervals) | Reaches 60 on paper, but only by counting motion, contact and appearance intervals that are not static anatomy; all unverified, 51 intervals in held-out. |
| Persistent vs transient | Cannot be defined until video length is known; durations are a weak proxy. |
| Verified negatives | **None.** The 10% false-positive gate cannot be evaluated until negatives are annotated. |
| Resolving a 10-point gain | With n held-out positives, a 95% interval on recall is about +/-0.26 (n=10), +/-0.20 (20), +/-0.17 (30), +/-0.12 (60), +/-0.10 (100). At the 14 or fewer held-out hand-anatomy intervals a 10-point difference between two methods is not resolvable even before verification losses. |
| VBench 142 test crops | Powers a **real-image feasibility signal**, not the temporal gate: roughly +/-0.11 on class recall at n of about 70 per class. |

Conclusion: Spotlight alone **underpowers** the preregistered temporal gate as written. Do not
lower the thresholds to compensate. VBench is enough for a first, honest real-image test of
the single-frame inspector, and only that.

## 8. Smallest next executable experiment

**ZM-HAND-03a: single-frame inspector on real generated hand crops (VBench-2.0).**

1. Fetch the 142 test crops (3.2 MB). Dry run first; nothing is downloaded without `--confirm`:
   ```powershell
   python -m examples.hand_anatomy_03.fetch_vbench_hand_test --output artifacts/zm_hand_03/vbench_hand_test
   python -m examples.hand_anatomy_03.fetch_vbench_hand_test --output artifacts/zm_hand_03/vbench_hand_test --confirm
   ```
2. **Resolve the mask blocker (decision needed).** Options, in the order I would take them:
   (a) a person draws masks for a stratified 40-image subset (20 normal / 20 abnormal), about
   1-2 hours, no new dependencies; (b) an authorized lightweight segmenter (not installed, so
   not assumed); (c) a classical skin/background threshold, reported as a segmentation-limited
   ceiling. Crops with a non-uniform natural background will not work with the existing
   `--segment-background` fallback, which refuses non-uniform borders by design.
3. Preregister before opening labels: the single-frame methods only (library IoU, refined IoU,
   and the ZM-HAND-02 residual score as an exploratory arm), thresholds set at 10% false
   positives on a video-level calibration subset, a video-level held-out test, and a stated
   no-effect outcome. Report class-wise recall with Wilson intervals; no temporal code.
4. Only if that shows measurable discrimination does a temporal study merit new annotation.

Optional parallel step for the Spotlight route: a human fills `manual_review_verdict`
(`hand_defect` / `not_hand` / `ambiguous`) in `artifacts/zm_hand_03/candidates.csv` for the 128 nominations plus the
63 manual-review rows, roughly an hour, before any video is downloaded. Candidate-clip download
(only after that review, and only the strict set of 35 clips, 199 MB):
```powershell
$files = Get-Content examples/hand_anatomy_03/data/spotlight_strict_hand_anatomy_clips.txt
huggingface-cli download UBC-ViL/Spotlight-VideoGen-Errors --repo-type dataset --local-dir artifacts/zm_hand_03/spotlight $files
```
(Requires `pip install huggingface_hub`; not installed or run here.)

## 9. Unresolved blockers

1. **No masks** for any real hand data (the critical one for ZM-HAND-01/02 real-image testing).
2. **No verified negatives** in Spotlight; none in VBench beyond its binary "normal" labels, whose
   definition is unconfirmed.
3. **Licensing ambiguity** for VBench (three statements, mixed-source contents) and
   non-commercial terms for Spotlight; neither is cleared for commercial filmmaking use.
4. Explicit extra/missing-digit defects are very rare in Spotlight (about 8 intervals).
5. VBench train/test leakage at the video level is unchecked; `hand_train.jsonl` was not read.
6. Crops lack wrist/forearm, mismatching the template family; video length is unknown for
   persistence analysis.

## Reproduction

```powershell
python -m examples.hand_anatomy_03.spotlight_inventory --metadata artifacts/zm_hand_03/raw/metadata.jsonl --output artifacts/zm_hand_03
python -m pytest examples/hand_anatomy_03/tests -q
```
`zip_tail_inventory.py` lists a zip's entries from a range-fetched tail without extracting anything.
