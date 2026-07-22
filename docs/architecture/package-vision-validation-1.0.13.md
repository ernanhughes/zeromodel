# ZeroModel 1.0.13 Vision Package Validation

Validation date: 2026-07-22

## Scope

`zeromodel-vision==1.0.13` is validated as the deterministic bounded
visual-address runtime package. It depends only on the validated core package,
the validated observation package, and NumPy at runtime.

This stage did not begin video isolation, SQLAlchemy isolation, cross-package
integration, benchmark materialization, mutation auditing, or scientific
threshold/provider selection.

## Validated Dependency Wheels

- `packages/core/dist/zeromodel-1.0.13-py3-none-any.whl`
- `packages/observation/dist/zeromodel_observation-1.0.13-py3-none-any.whl`
- `packages/vision/dist/zeromodel_vision-1.0.13-py3-none-any.whl`

Validated prior package commits:

- Core: `c827a36b6498990f2d7eb10e8ec4fc6a584fb502`
- Analysis: `e3ac0457e06bab9e7ae407ffa1d1fe1acbe9cabd`
- Observation: `0b6a4698633e55e99326488d4dbf77b1c266c560`

## Runtime Dependencies

`packages/vision/pyproject.toml` declares:

- `numpy>=1.23`
- `zeromodel==1.0.13`
- `zeromodel-observation==1.0.13`

Clean validation environment `pip freeze` included only pytest tooling plus
NumPy and the three local wheels. It did not install analysis, video,
SQLAlchemy, the repository root, Torch, TorchVision, Transformers, Pillow, or
Hugging Face packages.

## Production Public API

`zeromodel.vision.__all__` exports:

- `DISTANCE_METRIC`
- `MARGIN_RULE`
- `VISUAL_FEATURE_VERSION`
- `VISUAL_INDEX_VERSION`
- `VISUAL_POLICY_DECISION_VERSION`
- `VISUAL_READER_VERSION`
- `DeterministicVisualAddressProvider`
- `VisualDecision`
- `VisualFeatureSpec`
- `VisualIndexBuild`
- `VisualIndexCalibration`
- `VisualPolicyDecision`
- `VisualPolicyReader`
- `VisualSignReader`
- `build_visual_index`
- `extract_visual_features`
- `visual_feature_digest`
- `visual_input_digest`

Learned encoders, retrieval families, corruption sweeps, datasets,
registration experiments, and precomputed-provider helpers are not exported from
the production API.

## Schema Constants

- `VISUAL_FEATURE_VERSION = "zeromodel-visual-feature/v1"`
- `VISUAL_INDEX_VERSION = "zeromodel-visual-index/v2"`
- `VISUAL_READER_VERSION = "zeromodel-visual-sign-reader/v2"`
- `VISUAL_POLICY_DECISION_VERSION = "zeromodel-visual-policy-decision/v1"`
- `DISTANCE_METRIC = "euclidean"`
- `MARGIN_RULE = "absolute-gap"`

## Behavioral Coverage

Package-local tests cover:

- deterministic feature extraction from bounded image observations;
- immutable feature outputs and stable input/feature digests;
- deterministic visual index identity and calibration;
- incomplete and colliding codebook rejection;
- malformed threshold, margin, and closest-pair calibration rejection;
- canonical row recovery for all fixture rows;
- ambiguous near-tie rejection;
- far out-of-threshold rejection;
- observation-owned provider contract mapping;
- visual policy lookup round trip and digest reconstruction;
- mismatched address/policy contract rejection;
- import isolation from forbidden package families and heavy dependencies;
- wheel content isolation from copied root/core/observation/analysis/video files.

Canonical recovery result in the package-local fixture: `3` recovered rows,
`0` wrong rows. Ambiguous and far inputs are rejected with first-class decision
records instead of coerced policy actions.

`VisualPolicyDecision.from_dict()` reconstructs the observation-owned
`VisualAddressDecision` and core `PolicyLookupDecision` payloads, while
`VisualPolicyDecision.digest` preserves canonical decision identity over stable
JSON.

## Wheel Contents

The validated wheel contains only `zeromodel/vision/**` modules plus
`zeromodel_vision-1.0.13.dist-info/**` metadata:

- `zeromodel/vision/__init__.py`
- `zeromodel/vision/visual.py`
- `zeromodel/vision/visual_corruptions.py`
- `zeromodel/vision/visual_dataset.py`
- `zeromodel/vision/visual_encoder.py`
- `zeromodel/vision/visual_policy.py`
- `zeromodel/vision/visual_precomputed.py`
- `zeromodel/vision/visual_registration.py`
- `zeromodel/vision/visual_retrieval.py`

It does not include `zeromodel/__init__.py`, copied core modules, copied
observation modules, analysis modules, video modules, SQLAlchemy modules,
tests, docs, examples, scripts, or research files.

## Installed Import Paths

Clean wheel validation imported from:

- `C:\Projects\zeromodel\build\vision-isolation-venv\Lib\site-packages\zeromodel\core\__init__.py`
- `C:\Projects\zeromodel\build\vision-isolation-venv\Lib\site-packages\zeromodel\observation\__init__.py`
- `C:\Projects\zeromodel\build\vision-isolation-venv\Lib\site-packages\zeromodel\vision\__init__.py`

## Validation Commands

- `PYTHONPATH=packages/vision/src;packages/observation/src;packages/core/src python -m pytest -q packages/vision/tests`
  - Result: `10 passed in 0.33s`
- `python -m ruff check packages/vision/src packages/vision/tests`
  - Result: passed
- `python -m ruff format --check packages/vision/src packages/vision/tests`
  - Result: `10 files already formatted`
- `python -m mypy packages/vision/src`
  - Result: `Success: no issues found in 9 source files`
- `python scripts/check_package_boundaries.py`
  - Result: `Package boundary check passed: 118 production modules`
- `python scripts/check_quality.py`
  - Result: `Quality checks passed`
- `python -m build packages/core`
  - Result: built `zeromodel-1.0.13`
- `python -m build packages/observation`
  - Result: built `zeromodel_observation-1.0.13`
- `python -m build packages/vision`
  - Result: built `zeromodel_vision-1.0.13`
- `python -m twine check packages/core/dist/*`
  - Result: passed
- `python -m twine check packages/observation/dist/*`
  - Result: passed
- `python -m twine check packages/vision/dist/*`
  - Result: passed
- Clean venv `pip check`
  - Result: `No broken requirements found.`
- Clean venv installed-wheel tests with `VISION_WHEEL_PATH`
  - Result: `10 passed in 0.42s`

Build emitted setuptools deprecation warnings for existing license metadata.
Those warnings did not fail builds or `twine check`.

## Historical Tests

The root visual-address/sign-reader tests were replaced by package-local
contract tests under `packages/vision/tests`. The retained package tests avoid
examples, research outputs, external datasets, benchmark materialization, and
model/provider selection.

## Research Exclusions

This validation excludes:

- DINOv2/Hugging Face model behavior;
- learned embedding quality;
- local-correlation or discriminative evidence providers;
- corruption robustness claims;
- registration experiments;
- benchmark calibration and final split materialization;
- video temporal behavior;
- SQL persistence.

## Vision Closure

The follow-up closure gate classified and moved the historical helper modules
that were previously still present in the wheel.

| module | classification | required-by | shipped-in-wheel | reason |
|---|---|---|---|---|
| `zeromodel.vision.__init__` | public production runtime | package public API | yes | Exports deterministic visual-address API only. |
| `zeromodel.vision.visual` | public production runtime | deterministic visual index and sign reader | yes | Owns deterministic feature extraction, index identity, calibration, and visual address recovery. |
| `zeromodel.vision.visual_policy` | public production runtime | provider-neutral policy bridge | yes | Adapts deterministic visual addresses to observation contracts and core policy lookup. |
| `visual_corruptions` | research or benchmark implementation | visual robustness research | no | Moved to `research/visual/visual_corruptions.py`. |
| `visual_dataset` | research or benchmark implementation | visual benchmark datasets | no | Moved to `research/visual/visual_dataset.py`. |
| `visual_encoder` | research or benchmark implementation | learned encoder experiments | no | Moved to `research/visual/visual_encoder.py`. |
| `visual_precomputed` | research or benchmark implementation | approximate/precomputed provider experiments | no | Moved to `research/visual/visual_precomputed.py`. |
| `visual_registration` | research or benchmark implementation | registration and local-baseline experiments | no | Moved to `research/visual/visual_registration.py`. |
| `visual_retrieval` | research or benchmark implementation | vector retrieval and linear-probe experiments | no | Moved to `research/visual/visual_retrieval.py`. |

The rebuilt wheel contains only:

- `zeromodel/vision/__init__.py`
- `zeromodel/vision/visual.py`
- `zeromodel/vision/visual_policy.py`
- `zeromodel_vision-1.0.13.dist-info/**`

Closure validation results:

- Source tests: `10 passed in 0.57s`
- Installed-wheel tests: `10 passed in 0.34s`
- Ruff, format, mypy, package boundaries, quality, build, and `twine check`: passed

## Remaining Defects

No package-blocking defects remain for deterministic visual-address isolation.
The next structural stage is `zeromodel-video` isolation.
