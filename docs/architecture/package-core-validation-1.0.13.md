# ZeroModel 1.0.13 Core Package Validation

## Scope

This report covers only the `packages/core` validation stage. It does not validate
the analysis, observation, vision, video, or persistence distributions.

## Commits

- Baseline structural validation commit: `0db7089560390261da08d27c9d798cffabec4159`
- Core validation implementation commit: recorded by the commit containing this
  report.

## Core Modules

The core wheel owns these Python modules under the implicit `zeromodel`
namespace:

- `zeromodel.core.__init__`
- `zeromodel.core.artifact`
- `zeromodel.core.bundle`
- `zeromodel.core.content_identity`
- `zeromodel.core.lua`
- `zeromodel.core.matrix_blob`
- `zeromodel.core.metrics`
- `zeromodel.core.policy_lookup`
- `zeromodel.core.policy_transitions`
- `zeromodel.core.render`
- `zeromodel.core.views`

No `zeromodel/__init__.py` is present in the wheel.

## Public API

`zeromodel.core.__all__` is explicit and package-local:

- `BUNDLE_VERSION`
- `CANONICAL_METRICS`
- `LAYOUT_VERSION`
- `LayoutRecipe`
- `MANIFEST_NAME`
- `MATRIX_BLOB_VERSION`
- `MatrixBlob`
- `PNG_SIGNATURE`
- `POLICY_LUA_FORMAT`
- `POLICY_PLAN_VERSION`
- `POLICY_TRANSITION_EVIDENCE_VERSION`
- `POLICY_TRANSITION_SPEC_VERSION`
- `PROTOTYPE_UNIVERSE_IDENTITY_VERSION`
- `PolicyLookupDecision`
- `PolicyTransitionEvidence`
- `PolicyTransitionSpec`
- `PrototypeUniverseIdentity`
- `ROW_UNION_TRANSITION_SCOPE`
- `SPEC_VERSION`
- `ScoreTable`
- `SignReader`
- `UnresolvedArtifactIdentity`
- `VPMArtifact`
- `VPMCell`
- `VPMPolicyLookup`
- `VPMRegion`
- `VPMValidationError`
- `ViewProfile`
- `ViewSet`
- `array_content_digest`
- `as_field`
- `build_view`
- `build_views`
- `build_vpm`
- `bundle_manifest`
- `canonical_float64_bytes`
- `canonical_json_bytes`
- `compiled_plan_id`
- `from_bundle`
- `lua_policy_source`
- `metric_ids_for_rows`
- `pack_metrics`
- `png_bytes`
- `prototype_universe_identity`
- `score_table_from_metric_rows`
- `sha256_digest`
- `svg_text`
- `to_bundle`
- `to_uint8`
- `write_lua_policy`
- `write_png`
- `write_svg`

## Runtime Dependencies

The core runtime dependency set is:

- Python `>=3.10`
- `numpy>=1.23`

The clean validation environment additionally installed only test tooling:
`pytest`, `colorama`, `iniconfig`, `packaging`, `pluggy`, and `Pygments`.
`pip check` reported no broken requirements.

## Migrated Tests

Core-owned historical tests were relocated or rewritten as package-local tests:

- `tests/test_artifact_kernel.py` -> `packages/core/tests/test_artifact_kernel.py`
- `tests/test_lua_policy.py` -> `packages/core/tests/test_policy_lookup_lua.py`
- `tests/test_matrix_blob.py` -> `packages/core/tests/test_matrix_blob.py`
- `tests/test_policy_lookup.py` -> `packages/core/tests/test_policy_lookup_lua.py`
- `tests/test_policy_lookup_compiled.py` -> `packages/core/tests/test_policy_lookup_lua.py`
- `tests/test_policy_transitions.py` -> `packages/core/tests/test_policy_transitions.py`
- `tests/test_views.py` -> `packages/core/tests/test_views_bundle_render.py`

The package-local suite also adds explicit API, import-isolation, and wheel
content checks in `packages/core/tests/test_core_api_isolation_wheel.py`.

## Test Results

Source-tree validation:

```text
PYTHONPATH=packages/core/src python -m pytest -q packages/core/tests
20 passed, 1 skipped in 0.29s
```

Clean wheel validation:

```text
build/core-isolation-venv/Scripts/python.exe -m pytest -q packages/core/tests
20 passed, 1 skipped in 0.30s
```

The skipped test is optional Lua runtime execution when no Lua interpreter is
available. Lua source generation and compiled-plan identity are tested
unconditionally.

## Build And Wheel Contents

Build and metadata validation passed:

```text
python -m build packages/core
python -m twine check packages/core/dist/*
PASSED
```

The wheel contains only these entries:

- `zeromodel/core/__init__.py`
- `zeromodel/core/artifact.py`
- `zeromodel/core/bundle.py`
- `zeromodel/core/content_identity.py`
- `zeromodel/core/lua.py`
- `zeromodel/core/matrix_blob.py`
- `zeromodel/core/metrics.py`
- `zeromodel/core/policy_lookup.py`
- `zeromodel/core/policy_transitions.py`
- `zeromodel/core/render.py`
- `zeromodel/core/views.py`
- `zeromodel-1.0.13.dist-info/METADATA`
- `zeromodel-1.0.13.dist-info/WHEEL`
- `zeromodel-1.0.13.dist-info/top_level.txt`
- `zeromodel-1.0.13.dist-info/RECORD`

Rejected content assertions cover `zeromodel/__init__.py`, sibling ZeroModel
packages, root tests, research, examples, docs, and repository scripts.

## Clean Import Isolation

The installed import location resolves to the clean environment:

```text
C:\Projects\zeromodel\build\core-isolation-venv\Lib\site-packages\zeromodel\core\__init__.py
```

The import-isolation test verifies that importing `zeromodel.core` does not load
`sqlalchemy`, `torch`, `torchvision`, `transformers`, `PIL`,
`zeromodel.analysis`, `zeromodel.observation`, `zeromodel.vision`,
`zeromodel.video`, or `zeromodel.persistence`.

## Golden Identity Results

Golden identity behavior was preserved. The artifact-kernel test retains the
existing deterministic artifact ID:

```text
32f8013789e4ff463569e2ccbbdc8c3802bc42c6edeb8ceb361afca9a6025db1
```

MatrixBlob identity, metadata-sensitive identity, canonical native-endian float
identity, compiled-plan identity, bundle reconstruction, source digest
preservation, and Lua policy source determinism are covered by package-local
tests. No golden values were updated for this stage.

## Boundary And Quality

Package-boundary validation passed:

```text
python scripts/check_package_boundaries.py
Package boundary check passed: 118 production modules
```

Core quality checks passed:

```text
python -m ruff check packages/core/src packages/core/tests
All checks passed!

python -m ruff format --check packages/core/src packages/core/tests
17 files already formatted

python -m mypy packages/core/src
Success: no issues found in 11 source files
```

## Historical Test Classification

The root tests moved in this stage were classified as core contract tests. The
remaining historical root tests belong to later package-validation stages:
analysis, observation, vision, video, SQLAlchemy persistence, integration,
research, or repository tooling. They were not repaired or run as blockers for
core isolation.

## Deferred Issues

- Optional Lua execution remains skipped unless a Lua interpreter is available.
- `setuptools` emits deprecation warnings for the current license table and
  license classifier style during build. Metadata validation still passes; this
  can be modernized in a follow-up packaging cleanup.
- Analysis isolation is the next package-validation stage.
