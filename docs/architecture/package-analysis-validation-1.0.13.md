# ZeroModel 1.0.13 Analysis Package Validation

## Scope

This report covers only the `packages/analysis` validation stage. It does not
validate observation, vision, video, SQLAlchemy, or cross-package integration.

## Commits

- Validated core commit:
  `c827a36b6498990f2d7eb10e8ec4fc6a584fb502`
- Analysis validation implementation commit: recorded by the commit containing
  this report.

## Analysis Modules

The analysis wheel owns these modules under the implicit `zeromodel` namespace:

- `zeromodel.analysis.__init__`
- `zeromodel.analysis.adapters.__init__`
- `zeromodel.analysis.adapters.common`
- `zeromodel.analysis.adapters.jsonl`
- `zeromodel.analysis.adapters.tensorboard`
- `zeromodel.analysis.adapters.trackio`
- `zeromodel.analysis.adapters.wandb`
- `zeromodel.analysis.compare`
- `zeromodel.analysis.compose`
- `zeromodel.analysis.controller`
- `zeromodel.analysis.critic`
- `zeromodel.analysis.domains.__init__`
- `zeromodel.analysis.edge`
- `zeromodel.analysis.hierarchy`
- `zeromodel.analysis.learning`
- `zeromodel.analysis.manifold`
- `zeromodel.analysis.patterns`
- `zeromodel.analysis.phos`
- `zeromodel.analysis.policy_diagnostics`
- `zeromodel.analysis.policy_properties`
- `zeromodel.analysis.spatial`
- `zeromodel.analysis.training`

## Public API

`zeromodel.analysis.__all__` is explicit and analysis-owned. It intentionally
does not re-export core objects such as `ScoreTable` or `VPMArtifact`.

Public symbols:

- `CHECKER_VERSION`
- `CRITICALITY_METRIC_ID`
- `CriticAssessment`
- `CriticObservation`
- `DECISION_MARGIN_METRIC_ID`
- `Decision`
- `DecisionManifold`
- `HierarchyLevel`
- `LearningAssessment`
- `LearningObservation`
- `ManifoldFrame`
- `ManifoldSummary`
- `ManifoldTransition`
- `MatrixPatternDetector`
- `OBJECTIVE_IDS`
- `ObjectiveResult`
- `PATTERN_CHECKER_VERSION`
- `PATTERN_METHOD`
- `PHOSResult`
- `PatternAnalysisSpec`
- `PatternDiscoveryArtifacts`
- `PatternReport`
- `Policy`
- `PolicyPropertyChecker`
- `PolicyPropertyResult`
- `PolicyPropertySpec`
- `PolicyPropertyViolation`
- `PolicyVerificationReport`
- `REPORT_METRICS`
- `Signal`
- `SpatialOptimizationResult`
- `SpatialOptimizer`
- `TENSORBOARD_DEFAULT_ALIASES`
- `TRACKIO_DEFAULT_ALIASES`
- `Thresholds`
- `TopLeftGate`
- `TopLeftGateResult`
- `TrainingCheckpoint`
- `TrainingProgressAssessment`
- `VALID_SPLITS`
- `VERIFICATION_METRICS`
- `VPMComparison`
- `VPMController`
- `VPMRow`
- `WANDB_DEFAULT_ALIASES`
- `as_field`
- `build_critic_vpm`
- `build_decision_manifold`
- `build_discovered_view`
- `build_learning_vpm`
- `build_optimized_view`
- `build_pyramid`
- `build_training_progress_vpm`
- `checkpoints_from_csv`
- `checkpoints_from_export`
- `checkpoints_from_json`
- `checkpoints_from_jsonl`
- `checkpoints_from_tensorboard_scalars`
- `checkpoints_from_trackio_export`
- `checkpoints_from_wandb_export`
- `compare_fields`
- `critic_recipe`
- `decode_key_value_row_id`
- `default_controller`
- `detect_patterns`
- `find_inflection_points`
- `guarded_pack_artifact`
- `image_entropy`
- `learning_recipe`
- `load_tracker_records`
- `observations_from_critic_lines`
- `optimize_view_profile`
- `pack_artifact`
- `phos_sort_pack`
- `records_to_checkpoints`
- `reduce_blocks`
- `robust01`
- `to_square`
- `top_left_concentration`
- `training_progress_recipe`
- `vpm_add`
- `vpm_and`
- `vpm_not`
- `vpm_or`
- `vpm_subtract`
- `vpm_xor`
- `with_q_diagnostics`

## Runtime Dependencies

The analysis runtime dependency set is:

- Python `>=3.10`
- `numpy>=1.23`
- `zeromodel==1.0.13`

The core wheel used for validation was:

```text
packages/core/dist/zeromodel-1.0.13-py3-none-any.whl
```

## Migrated Tests

Analysis-owned historical tests were moved or rewritten as package-local tests:

- `tests/test_blog_capabilities.py` ->
  `packages/analysis/tests/test_analysis_capabilities.py`
- `tests/test_critic.py` -> `packages/analysis/tests/test_critic.py`
- `tests/test_learning.py` -> `packages/analysis/tests/test_learning.py`
- `tests/test_manifold.py` -> `packages/analysis/tests/test_manifold.py`
- `tests/test_patterns.py` -> `packages/analysis/tests/test_patterns.py`
- `tests/test_phos.py` -> `packages/analysis/tests/test_phos.py`
- `tests/test_policy_diagnostics.py` ->
  `packages/analysis/tests/test_policy_diagnostics.py`
- `tests/test_policy_properties.py` ->
  `packages/analysis/tests/test_policy_properties.py`
- `tests/test_spatial.py` -> `packages/analysis/tests/test_spatial.py`
- `tests/test_training.py` -> `packages/analysis/tests/test_training.py`
- `tests/test_training_adapters.py` ->
  `packages/analysis/tests/test_training_adapters.py`

`packages/analysis/tests/test_analysis_api_isolation_wheel.py` adds explicit
public API, forbidden-import, and wheel-content checks.

## Test Results

Source-tree validation:

```text
PYTHONPATH=packages/analysis/src;packages/core/src python -m pytest -q packages/analysis/tests
65 passed in 0.82s
```

Clean wheel validation:

```text
ANALYSIS_WHEEL_PATH=packages/analysis/dist/zeromodel_analysis-1.0.13-py3-none-any.whl `
  build/analysis-isolation-venv/Scripts/python.exe -m pytest -q packages/analysis/tests
65 passed in 1.19s
```

`pip check` passed in the clean environment.

## Import Locations

Both packages resolved inside the clean validation environment:

```text
C:\Projects\zeromodel\build\analysis-isolation-venv\Lib\site-packages\zeromodel\core\__init__.py
C:\Projects\zeromodel\build\analysis-isolation-venv\Lib\site-packages\zeromodel\analysis\__init__.py
```

The forbidden-import test verifies that importing `zeromodel.analysis` loads
`zeromodel.core` and does not load `zeromodel.observation`, `zeromodel.vision`,
`zeromodel.video`, `zeromodel.persistence`, `sqlalchemy`, `torch`,
`torchvision`, `transformers`, or `PIL`.

## Wheel Contents

The analysis wheel contains only these entries:

- `zeromodel/analysis/__init__.py`
- `zeromodel/analysis/compare.py`
- `zeromodel/analysis/compose.py`
- `zeromodel/analysis/controller.py`
- `zeromodel/analysis/critic.py`
- `zeromodel/analysis/edge.py`
- `zeromodel/analysis/hierarchy.py`
- `zeromodel/analysis/learning.py`
- `zeromodel/analysis/manifold.py`
- `zeromodel/analysis/patterns.py`
- `zeromodel/analysis/phos.py`
- `zeromodel/analysis/policy_diagnostics.py`
- `zeromodel/analysis/policy_properties.py`
- `zeromodel/analysis/spatial.py`
- `zeromodel/analysis/training.py`
- `zeromodel/analysis/adapters/__init__.py`
- `zeromodel/analysis/adapters/common.py`
- `zeromodel/analysis/adapters/jsonl.py`
- `zeromodel/analysis/adapters/tensorboard.py`
- `zeromodel/analysis/adapters/trackio.py`
- `zeromodel/analysis/adapters/wandb.py`
- `zeromodel/analysis/domains/__init__.py`
- `zeromodel_analysis-1.0.13.dist-info/METADATA`
- `zeromodel_analysis-1.0.13.dist-info/WHEEL`
- `zeromodel_analysis-1.0.13.dist-info/top_level.txt`
- `zeromodel_analysis-1.0.13.dist-info/RECORD`

Rejected content assertions cover `zeromodel/__init__.py`, copied core files,
sibling ZeroModel packages, root tests, research, examples, docs, and scripts.

## Identity And Schema Results

Analysis schema/version constants are preserved:

- `CHECKER_VERSION`
- `PATTERN_CHECKER_VERSION`
- `PATTERN_METHOD`
- `POLICY_TRANSITION_EVIDENCE_VERSION` and
  `POLICY_TRANSITION_SPEC_VERSION` remain core-owned and are consumed from core.

Package-local tests preserve deterministic report and artifact identities for:
pattern reports, discovered views, policy verification reports, PHOS packing,
decision manifolds, critic artifacts, learning artifacts, and training-progress
artifacts. No golden value was updated for this stage.

## Boundary And Quality

Package-boundary validation passed:

```text
python scripts/check_package_boundaries.py
Package boundary check passed: 118 production modules
```

Analysis quality checks passed:

```text
python -m ruff check packages/analysis/src packages/analysis/tests
python -m ruff format --check packages/analysis/src packages/analysis/tests
python -m mypy packages/analysis/src
```

Repository quality tooling was made workspace-aware and passed for the core and
analysis package scope:

```text
python scripts/check_quality.py
Quality checks passed
```

The migrated legacy quality ceilings for core Lua, core policy lookup, and
analysis training are recorded in `quality-baseline.toml`.

## Historical Test Classification

The tests listed above were moved because they validate analysis-owned behavior.
Remaining root tests are classified for later stages: observation, vision,
video, SQLAlchemy persistence, research, examples, repository tooling, or
integration. They were not run as blockers for analysis isolation.

## Research Relocations

No research-only code was found inside `packages/analysis/src`, and no production
analysis module imports `research`.

## Deferred Issues

- The `setuptools` license-table and license-classifier deprecation warnings
  remain from the package metadata style and can be modernized later.
- Full-repository quality reporting still exposes legacy exceptions outside the
  core and analysis validation scope; later package stages should reconcile
  those paths as their packages are validated.
- Observation isolation is the next package-validation stage.
