# ZM-HAND-01 — Visual Anatomical Inspection (experiment)

An **unpromoted example**, not a production package. It tests one hypothesis:

> Anatomically valid hand structures, compiled into an identified template
> library and compared against an observed silhouette through a genuine
> ZeroModel VPM, can expose observed structure that no permitted anatomical
> configuration explains.

Results (including negative ones) are in [RESULTS.md](RESULTS.md). Nothing here
is validated on real images.

## Design note — what is reused

| Need | Reused from the repo | Where |
|---|---|---|
| Score matrix with stable ids | `zeromodel.core.ScoreTable` | `vpm_compiler.py` |
| Explicit ordering/normalisation | `zeromodel.core.LayoutRecipe` (`weighted_score` rows, explicit columns, `per_metric_minmax`) | `vpm_compiler.py` |
| Identity-bearing artifact | `zeromodel.core.build_vpm` → `VPMArtifact` (`artifact_id`, `from_dict` re-validation) | `vpm_compiler.py`, `evidence.py` |
| Canonical digests | `zeromodel.core.content_identity` (`sha256_digest`, `array_content_digest`) | `_common.py` |
| Core PNG of a VPM | `zeromodel.core.render.png_bytes` | `evidence.py` |

Not changed: `zeromodel.core`, `zeromodel.vision`, any package, `package-boundaries.toml`,
`VERSION`. Only `numpy` and `Pillow` (both already dev dependencies) are used. No
`zeromodel.vision` / `perception` / `critic` dependency was needed. APIs matched the
prompt's description except: `LayoutRecipe` needs `version: "vpm-layout/0"` and the
only supported normalisation is per-metric min-max, which is why every decision reads
**raw** cells (`VPMArtifact.cell().raw_value`), never the normalised view.

## Data flow

```
HandPose (validated joint ranges) --build_skeleton--> 3D capsule skeleton (21 landmarks)
  --orthographic projection + painter's order--> labelled 2D template (+ geometry_id / render_id / mask_id)
  --> TemplateLibrary (96 templates, library_id)                         [geometry.py, template_renderer.py, pose_library.py]

image + mask --> Observation (original preserved; separate normalised work mask; observation_id)   [observations.py]
  --> bounded alignment (rot +/-45, residual scale, translation; deterministic ties)               [alignment.py]
  --> independent metrics per candidate (IoU, chamfer, geometry, unexplained, protrusion,
      missing, structure; optional landmarks)                                                      [comparison.py]
  --> ScoreTable(rows=templates, cols=metrics) --LayoutRecipe--> build_vpm --> VPMArtifact          [vpm_compiler.py]
  --> decision contract reads RAW VPM cells --> supported_plausible / suspected_anomaly /
      insufficient_evidence / unsupported_input                                                    [assessment.py]
  --> report.json (embeds the VPM, bindings, limitations) + overlays; validate / replay             [evidence.py, visualization.py]
```

Key principle (`assessment.py`): the verdict is **not** "which template looks most similar".
It evaluates the best-case *explainer* among permitted candidates (IoU above `fit_floor`)
and flags structure that even that explainer leaves unexplained. A hard contradiction is
never averaged away by high overlap. Missing structure is never used to claim an anomaly
(ambiguous with occlusion). Thresholds are exploratory until calibrated; the benchmark
fits them on a calibration split only and writes `calibrated_contract.json`.

## Commands (Windows PowerShell, from the repository root)

No install is needed (`_bootstrap.py` adds `packages/*/src` to `sys.path` if
`zeromodel` is not installed). Outputs go to the git-ignored `artifacts/`.

```powershell
python -m examples.hand_anatomy.cli build-library --output artifacts/hand_anatomy/library
python -m examples.hand_anatomy.cli verify-library --library artifacts/hand_anatomy/library
python -m examples.hand_anatomy.cli inspect-library --library artifacts/hand_anatomy/library
python -m examples.hand_anatomy.cli contact-sheet --library artifacts/hand_anatomy/library --output artifacts/hand_anatomy/sheet.png

python -m examples.hand_anatomy.cli benchmark --output artifacts/hand_anatomy/benchmark      # ~90 s, 4 workers
python -m examples.hand_anatomy.cli make-fixtures --output artifacts/hand_anatomy/fixtures

# Case A (plausible) and Case B (six-finger); fixtures come from the independent generator
python -m examples.hand_anatomy.cli inspect `
  --mask artifacts/hand_anatomy/fixtures/plausible_open_hand-mask.png `
  --library artifacts/hand_anatomy/library `
  --contract artifacts/hand_anatomy/benchmark/calibrated_contract.json `
  --output artifacts/hand_anatomy/inspection/plausible
python -m examples.hand_anatomy.cli inspect `
  --mask artifacts/hand_anatomy/fixtures/six_finger_hand-mask.png `
  --library artifacts/hand_anatomy/library `
  --contract artifacts/hand_anatomy/benchmark/calibrated_contract.json `
  --output artifacts/hand_anatomy/inspection/six_finger

python -m examples.hand_anatomy.cli validate-report --report artifacts/hand_anatomy/inspection/six_finger/report.json
python -m examples.hand_anatomy.cli replay --report artifacts/hand_anatomy/inspection/six_finger/report.json `
  --mask artifacts/hand_anatomy/fixtures/six_finger_hand-mask.png --library artifacts/hand_anatomy/library
```

Your own input: `inspect --image hand.png --mask hand-mask.png` (white = hand, same size),
or an RGBA image whose alpha is the mask, or `--segment-background` (only valid for a
near-uniform background; it refuses otherwise). Optional `--landmarks landmarks.json`
(21 `[x, y]` pairs from an external provider) adds a `landmark_compatibility` column.
Exit codes: 0 ok, 1 validation/replay mismatch, 2 unsupported or invalid input.

Tests (marked `research`, so outside the fast gate by design):

```powershell
python -m pytest examples/hand_anatomy/tests -q
```

## Overlay colours

| Colour | Meaning |
|---|---|
| green | observed structure explained by the aligned template |
| red | observed structure **not** explained by the template (beyond a 2 px tolerance) |
| blue | structure the template expects but the observation lacks (may be occlusion) |
| grey | observed pixels inside the tolerance band only |

Inspection output files: `01_original`, `02_normalized_mask`, `03_best_template`,
`04_overlay_best/explainer`, `05_unexplained_observed`, `06_missing_expected`,
`07_candidate_contact_sheet`, `08_vpm_annotated` (grey = normalised view, text = raw
value), `08_vpm_core_render`, `report.json`. All are derived from the alignment recorded
in the report.

## Declared limitations

* Orthographic capsule skeleton, 3 yaw angles, 2 hands, 8 gestures, 2 proportion sets
  = 96 templates. Not a mesh, not MANO (no MANO asset is used or downloaded), no
  Blender path. The provider interface for true 3D geometry is **not** implemented.
* A single silhouette cannot give depth, palm-vs-back facing, handedness (a left hand
  seen palm-first and a right hand seen back-first share a silhouette up to mirroring),
  or occlusion order. Handedness is therefore never used in a decision.
* In-plane rotation search is bounded to +/-45 degrees.
* Mask quality is assumed. Segmentation errors can look like anomalies.
* The finger-like-projection count is a heuristic that fails under overlap; it is
  disabled by the calibrated contract.
* The VPM view is min-max normalised per column by the core kernel; decisions use raw cells.
* Landmark support is implemented and unit-checked but **not** benchmarked.
