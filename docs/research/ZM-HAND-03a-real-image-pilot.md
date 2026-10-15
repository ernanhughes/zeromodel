# ZM-HAND-03a — Real-image development pilot

Status: **development gate not passed. Stopped before any held-out data was fetched.**
ZM-HAND-01 and ZM-HAND-02 results are unchanged (see "Changes to frozen code" below).
All numbers come from 40 development crops, so they are a pilot, not a performance estimate.

## Research decision: INPUT INCOMPATIBILITY

Real-image representation and segmentation are too unreliable to test the anatomical
hypothesis fairly with the current method. Three independent observations support this:

1. **The masks mostly do not resolve fingers.** Of 40 development crops, 13 failed
   segmentation outright; of 14 masks reviewed blind, only 4 were judged good or acceptable.
2. **A good template fit is not evidence of a hand.** Non-hand blobs fit the forearm-free
   templates as well as real hand crops do (table below).
3. **No method discriminates.** Among 27 eligible crops, every score (library IoU, refined
   IoU, residual, branch, protrusion) has AUC between 0.43 and 0.54, i.e. chance.

This does not show that anatomical geometry is useless; it shows this silhouette pipeline cannot
be evaluated on these crops, because the crops and masks do not carry the structure it compares.

## Stage 1: split integrity (the published split is not video-disjoint)

Labels were obtained by range-reading `hand_train.jsonl` (775 KB compressed) from the split zip.

| Check | Result |
|---|---|
| Train rows / test rows | 39,848 / 142 |
| Train labels | 21,833 normal, 18,015 abnormal |
| Distinct source-video ids | 10,978 train, 77 test |
| Test video ids also in train | **74 of 77** |
| Test crops whose video is in train | **139 of 142** |
| Filenames present in both files | **22** |

The published test file is therefore **not an independent video-level test**, and the original
142 crops were **not used**. A video-disjoint partition was built over the pooled labels:
videos ranked by SHA-256 of `"zm-hand-03a:" + video_id`; first 5% development pool (550 videos),
last 50% held-out pool (5,490), the middle unused. Selection uses only label, video id and
hashes (never pixels): 20 normal + 20 abnormal, one crop per video, from the development pool,
none from an original-test video. Saved as `examples/hand_anatomy_03/data/vbench_dev_selection.json`.
The held-out pool was never opened.

## Stage 2: development crops (40 images, 528 KB, CRC-verified)

Fetched by range request from `opensource.z01` only. See `artifacts/zm_hand_03/dev_contact_sheet.png`.

* **Size:** 20x20 to 763x762 px; 4 crops are under 32 px on the short side.
* **Confound:** abnormal crops are smaller (median short side 68 px) than normal crops
  (126 px). An image-size score alone gives AUC 0.64 over all 40. Any real-image result must be
  checked against it.
* **Wrist/forearm:** almost none of the crops show a wrist or forearm.
* **Hands:** mostly close-ups in contact with objects, often blurred; many fingers occluded or
  not individually identifiable; some crops are multiple or merged hands.
* **Detector false positives:** at least one "abnormal" crop is a person in a dress, not a hand;
  two are tiny background patches. "Abnormal" therefore includes bad detections and low quality,
  not only anatomical defects.
* **Real vs generated:** cannot be told from filenames (hashes do not match the archive's
  source-video names) or reliably by eye. The domain is unknown.

## Stage 3: segmenter

* **Provider:** U^2-Net-small (`u2netp.onnx`), checkpoint sha256
  `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`, Apache-2.0 (U^2-Net;
  ONNX export distributed with rembg, MIT). 4.6 MB, CPU, about 0.5 s per crop.
* **Isolation:** its own virtual environment (`onnxruntime`, `numpy`, `pillow`); the script
  `segment_u2netp.py` imports nothing from ZeroModel. Preprocessing and thresholds are recorded in
  `segmentation.json`.
* **Per-crop output:** mask, probability map, overlay, explicit status
  (`ok` / `low_confidence` / `failed`) with reasons, and quality evidence.

### Does the segmenter preserve a known extra finger? (controlled test, rule declared first)

19 ZM-HAND-02 extra-finger pairs rendered as skin-toned images with a textured background.

| Condition | Median IoU (defect image) | Extra-region recall (median) | Images with recall >= 0.8 |
|---|---|---|---|
| Easy (contrasting background) | 0.96 | 1.00 | 100% |
| Hard (skin-like background) | 0.26 | 0.45 | 42% |

By the declared rule the segmenter is **suitable only for the contrasting-background condition**
and **unsuitable under skin-like backgrounds**, which many real crops resemble. It does not hide
a known extra finger when contrast is good, which is the useful half of the finding.

### Real-crop mask quality

Status over 40 crops: 18 ok, 9 low_confidence (kept, flagged), **13 failed** (7 mask area under 2%,
4 image under 32 px, 2 masks below the 400 px minimum used by the comparison code). A separate
reviewer, blind to labels and statuses, rated 14 random original/overlay pairs:

| Reviewer rating | Count |
|---|---|
| mask good | 1 |
| mask acceptable | 3 |
| mask poor (wrong object, empty, or fingers unresolved) | 10 |
| no recognisable hand in the original | 3 |

Of the 7 crops the pipeline marked `ok`, the reviewer judged 5 poor, so the pipeline's own status
does not catch bad masks. The reviewer was a model, not a human; it is independent of the
pipeline and the labels, but a human spot-check remains advisable. Typical failures: skin-on-skin
crops, the handle or dress selected instead of the hand, a hand merged with an object.

## Stage 4-5: comparison on real crops

Representation changes, separate from the frozen originals (`real_pipeline.py`):
forearm-free templates (new identities), the same refinement with the forearm dropped, four
quadrant rotations on top of the +/-45 degree search, border contact as a covariate. Eligibility
used input quality only: not segmentation-failed and at least 400 mask pixels.

| Quantity | Result |
|---|---|
| Eligible crops | 27 of 40 (17 normal, 10 abnormal) |
| Library IoU, eligible | median 0.77 (0.67-0.81) |
| Refined IoU, eligible | median 0.83; all 27 at least 0.5, 26 at least 0.7 |
| Border contact above 10% | 15 of 27 |
| Time | median 9 s per crop (4 rotations x pipeline) |

| Score | AUC abnormal vs normal (10 vs 17) |
|---|---|
| Library IoU (A_iou) | 0.535 |
| Refined IoU (A_refined_iou) | 0.512 |
| ZM-HAND-02 residual (C_residual) | 0.518 |
| ZM-HAND-01 protrusion | 0.497 |
| Residual, unrefined | 0.491 |
| Branch count | 0.432 |
| *Confound: smaller image is abnormal* | 0.544 (eligible), 0.644 (all 40) |
| *Confound: low segmentation confidence is abnormal* | 0.565 |

With 10 and 17 crops these AUCs have intervals of roughly +/-0.2, so they say "no detectable
signal", not "exactly chance".

**Non-hand control.** The same pipeline on 12 synthetic non-hand shapes (ellipses, rounded
rectangles, L-shapes, blob clusters) gives library IoU median **0.80** (0.65-0.82) and refined IoU
median **0.83** (0.71-0.86), versus **0.77** and **0.83** for the real crops. A template fit of this
size therefore does not indicate a hand: the comparison is uninformative on these masks. This is
the main reason for the INPUT INCOMPATIBILITY verdict.

Overlays: `artifacts/zm_hand_03/pilot/pilot_comparison_sheet.png`.

## Stage 6-7: not run

The continuation condition (the pipeline can represent real images without destroying relevant
anatomical evidence) was not met, so no held-out set was fetched, no thresholds were frozen, and no
held-out preregistration was written. Writing one now would only schedule a test already predicted
to be uninformative.

## What the pilot does not establish

* Not that the VBench abnormal label is mostly anatomical: it appears to include detector errors and
  low-quality crops.
* Not that other segmenters or higher-resolution hands would fail.
* Not that internal-structure defects are undetectable; silhouettes simply cannot show them.
  Your remark applies: a fused-finger hand can have a plausible outline.

## Recommendation for the next experiment

Do not run the held-out evaluation or build the temporal system on these crops. The smallest useful
next step is on the **input side**, not the method:

1. Obtain hands where finger structure is actually resolvable: sufficiently large (for example at
   least 128 px), with a wrist, little blur and little occlusion. Filter the VBench pool by size and
   a human-checked quality tag *before* any detector or label is involved.
2. Add a hand-landmark or part-level representation (21-point landmarks, or per-finger regions) as
   the second representation instead of more silhouette templates, and test it first on the
   ZM-HAND-02 pairs to see whether it separates a valid hand from its own defective twin.
3. Keep the non-hand control and the image-size confound baseline as mandatory comparisons.

## Changes to frozen code (additive, default off)

`examples/hand_anatomy_02/refine.py` and `inspect02.py` gained an optional `drop_forearm=False`
argument for the crop-compatible variant. With the default the behaviour is identical:
the 22 ZM-HAND-02 tests pass and re-evaluating the saved per-case records reproduces the frozen
`results.json` evaluation exactly (verdict REJECT).

## Downloads (all bounded)

| Item | Size |
|---|---|
| `hand_train.jsonl` range read | 0.78 MB |
| 40 development crops (range reads) | 0.53 MB |
| `u2netp.onnx` (rembg release asset) | 4.6 MB |
| `onnxruntime`, `numpy`, `pillow` into an isolated venv | pip packages |

Nothing from the held-out pool and none of the 142 original test crops was fetched. Raw and
derived files are under git-ignored `artifacts/zm_hand_03/`.

## Reproduce

```powershell
# one-time isolated segmenter environment
python -m venv artifacts/zm_hand_03/venv_seg
artifacts/zm_hand_03/venv_seg/Scripts/python.exe -m pip install onnxruntime numpy pillow
# (place u2netp.onnx in artifacts/zm_hand_03/seg/; sha256 is verified by the script)

python -m examples.hand_anatomy_03.prepare_dev
artifacts/zm_hand_03/venv_seg/Scripts/python.exe examples/hand_anatomy_03/segment_u2netp.py `
  --images artifacts/zm_hand_03/vbench_dev --output artifacts/zm_hand_03/seg/dev --model artifacts/zm_hand_03/seg/u2netp.onnx
python -m examples.hand_anatomy_03.segmenter_preservation_test --output artifacts/zm_hand_03/preservation
python -m examples.hand_anatomy_03.pilot_cli
python -m examples.hand_anatomy_03.blob_control
python -m pytest examples/hand_anatomy_03/tests -q
```
