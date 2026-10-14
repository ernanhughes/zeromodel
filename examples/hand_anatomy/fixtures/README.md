# Fixtures

No image files are committed: the repository `.gitignore` excludes `*.png`, and the
benchmark cases are fully reproducible from code.

* `python -m examples.hand_anatomy.cli make-fixtures --output <dir>` writes two demo
  masks (`plausible_open_hand-mask.png`, `six_finger_hand-mask.png`) from the
  **independent** blob generator in `synthetic.py` (not the template renderer).
* Benchmark cases (`synthetic.build_cases()`) are deterministic: ids are digests of the
  generating parameters; the calibration and test splits use disjoint seed spaces.

## Format for annotated external images (not yet used for any result)

One folder per case:

```
case-001/image.png        RGB/RGBA crop of one hand (PNG/JPEG/BMP, <= 2048 px a side)
case-001/mask.png         binary mask, white = hand, same size as image.png
case-001/label.json       {"label": "valid" | "malformed", "defect": "extra_digit" | "missing_digit" |
                           "abnormal_wrist" | "distorted_digit" | "other" | null,
                           "ambiguous": false, "source": "<generator/model name>", "notes": ""}
case-001/landmarks.json   optional {"landmarks": [[x, y] * 21]} from an external provider
```

Labels must be assigned by a human **before** looking at the inspector's output, and the
set must be split into calibration and held-out test sets before any threshold is chosen.
Import and score with the same calibrated contract and IoU baseline threshold:

```powershell
python -m examples.hand_anatomy.cli evaluate-external --root path/to/cases `
  --calibration artifacts/hand_anatomy/benchmark/calibrated_contract.json `
  --output artifacts/hand_anatomy/external
```

Do not reuse real cases to re-fit thresholds and then report the same cases as test results.
