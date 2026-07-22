# ZeroModel 1.0.13 Video Package Validation

Validation date: 2026-07-22

Baseline validated commits:

- Core: `c827a36b6498990f2d7eb10e8ec4fc6a584fb502`
- Analysis: `e3ac0457e06bab9e7ae407ffa1d1fe1acbe9cabd`
- Observation: `0b6a4698633e55e99326488d4dbf77b1c266c560`
- Vision before closure: `9f69e4d1319dfd3a6562fb47da5a2485a9058148`

Final commit: the commit containing this report; exact SHA is reported by
`git rev-parse HEAD` after commit creation.

## Scope

`zeromodel-video==1.0.13` is validated as an independently buildable,
installable, and testable video runtime package. It depends only on core,
observation, and NumPy at runtime. It does not depend on analysis, vision,
SQLAlchemy, persistence, research, examples, Torch, TorchVision, Transformers,
or Pillow.

SQLAlchemy isolation and final cross-package integration were not started.

## Module List

The source package contains 60 Python modules under `zeromodel/video/**`.
The wheel contains those 60 modules plus `zeromodel_video-1.0.13.dist-info/**`.

Major groups:

- frame and clip runtime: `video.py`
- provider-neutral temporal policy: `video_policy.py`
- reachability tile helpers: `video_policy_reachability.py`
- default runtime composition: `runtime.py`
- in-memory Store: `stores/video_action_set_memory.py`
- video action-set DTOs, services, facades, engines, planning, provenance, and
  final-access DTO/service modules under `domains/video_action_set/**`
- deterministic arcade policy mechanics under `arcade_policy/**`

## Public Exports

`zeromodel.video.__all__` exports only:

- version constants: `VIDEO_CLIP_MANIFEST_VERSION`,
  `VIDEO_FRAME_SOURCE_VERSION`, `VIDEO_FRAME_VERSION`,
  `VIDEO_POLICY_DECISION_VERSION`, `VIDEO_POLICY_TRACE_VERSION`,
  `VIDEO_TEMPORAL_EVIDENCE_VERSION`
- frame and clip contracts: `VideoFrame`, `VideoFrameSource`,
  `InMemoryVideoFrameSource`, `VideoClipManifest`
- temporal policy: `TemporalEvidence`, `VideoPolicyDecision`,
  `VideoPolicyReader`, `VideoPolicyTrace`
- action-set DTOs and services: `BenchmarkIdentityDTO`, `CanonicalJsonDTO`,
  `EpisodeCountsDTO`, `EpisodeIdsByFamilyDTO`, `EpisodePlanDTO`,
  `EpisodePlanService`, `MaterializedObservationDTO`, `ObservationDTO`,
  `ObservationOperationChainDTO`, `ObservationOperationDTO`,
  `ObservationService`, `ProviderObservationDescriptorDTO`,
  `SealedSplitPlanDTO`, `VideoActionSetFacade`, `VideoActionSetStore`
- in-memory/default runtime: `InMemoryVideoActionSetStore`,
  `ZeroModelRuntime`, `build_runtime`

It does not re-export core, observation contracts, concrete vision providers,
SQL Stores, research orchestration, or old root compatibility aliases.

## Runtime Dependencies

`packages/video/pyproject.toml` declares:

- `numpy>=1.23`
- `zeromodel==1.0.13`
- `zeromodel-observation==1.0.13`

Clean validation installed only pytest tooling, NumPy, core, observation, and
video. It did not install vision or SQLAlchemy.

## Wheels Used

- `packages/core/dist/zeromodel-1.0.13-py3-none-any.whl`
- `packages/observation/dist/zeromodel_observation-1.0.13-py3-none-any.whl`
- `packages/video/dist/zeromodel_video-1.0.13-py3-none-any.whl`

## Schema Constants

- `VIDEO_FRAME_VERSION = "zeromodel-video-frame/v1"`
- `VIDEO_CLIP_MANIFEST_VERSION = "zeromodel-video-clip-manifest/v1"`
- `VIDEO_FRAME_SOURCE_VERSION = "zeromodel-video-frame-source/v1"`
- `VIDEO_TEMPORAL_EVIDENCE_VERSION = "zeromodel-video-temporal-evidence/v1"`
- `VIDEO_POLICY_DECISION_VERSION = "zeromodel-video-policy-decision/v1"`
- `VIDEO_POLICY_TRACE_VERSION = "zeromodel-video-policy-trace/v1"`

Video action-set constants remain in
`zeromodel.video.domains.video_action_set.contracts`.

## Preserved Identities

Tests preserve deterministic frame digests, pixel digests, clip manifest IDs,
trace IDs, benchmark seed digests, episode plan digests, sealed plan digests,
and MatrixBlob content identity after package isolation. Identity-bearing bytes
do not include repository paths, module paths, object memory addresses, database
row IDs, or SQL table names.

## Test Coverage

Package-local tests:

- `test_frame_and_clip.py`
- `test_temporal_policy.py`
- `test_action_set_store.py`
- `test_import_isolation_and_wheel.py`

Covered behavior:

- frame identity, caller-memory isolation, dtype/shape validation, pixel digest
  preservation, clip ordering, manifest identity, duplicate/empty rejection
- provider-neutral temporal policy using only observation contracts, accepted
  frames, rejected provider outcomes, action transitions, repeated actions,
  unknown policy row rejection, policy manifest mismatch, trace determinism
- benchmark identity writes, episode plan writes, sealed final split plan
  plan-only/materialization-prohibited semantics, MatrixBlob write/read,
  idempotent writes, conflicting write rejection, batch atomicity
- default runtime using in-memory Stores without SQLAlchemy
- import isolation from forbidden/heavy dependencies
- wheel content isolation

## Validation Results

- Source package tests: `10 passed in 0.69s`
- Installed-wheel tests: `10 passed in 0.44s`
- `python -m ruff check packages/video/src packages/video/tests`: passed
- `python -m ruff format --check packages/video/src packages/video/tests`: passed
- `python -m mypy packages/video/src`: `Success: no issues found in 60 source files`
- `python scripts/check_package_boundaries.py`: `Package boundary check passed: 112 production modules`
- `python scripts/check_quality.py`: `Quality checks passed`
- `python -m build packages/core`: passed
- `python -m build packages/observation`: passed
- `python -m build packages/video`: passed
- `python -m twine check packages/core/dist/*`: passed
- `python -m twine check packages/observation/dist/*`: passed
- `python -m twine check packages/video/dist/*`: passed
- Clean venv `pip check`: `No broken requirements found.`

Build emitted existing setuptools deprecation warnings for license metadata.
Those warnings did not fail builds or `twine check`.

## Clean Import Paths

Clean wheel validation imported from:

- `C:\Projects\zeromodel\build\video-isolation-venv\Lib\site-packages\zeromodel\core\__init__.py`
- `C:\Projects\zeromodel\build\video-isolation-venv\Lib\site-packages\zeromodel\observation\__init__.py`
- `C:\Projects\zeromodel\build\video-isolation-venv\Lib\site-packages\zeromodel\video\__init__.py`

## Forbidden Imports

Subprocess import tests confirmed importing `zeromodel.video` does not import:

- `zeromodel.analysis`
- `zeromodel.vision`
- `zeromodel.persistence`
- `sqlalchemy`
- `torch`
- `torchvision`
- `transformers`
- `PIL`
- `research`

## Wheel Contents

The wheel contains only:

- `zeromodel/video/**`
- `zeromodel_video-1.0.13.dist-info/**`

It does not contain copied root namespace files, core modules, observation
modules, analysis modules, vision modules, persistence modules, tests, research,
examples, docs, scripts, database files, benchmark results, or research
evidence.

## CLI Classification

No video package console scripts are declared in `packages/video/pyproject.toml`.
Historical benchmark, provider-comparison, and evidence-generation CLIs remain
outside the video package or are classified for research/integration stages.

## Historical Tests

New package-local tests were added for the independently supported video runtime
surface. Large historical root tests that exercise visual providers, research
benchmarks, SQL stores, final execution, or cross-package integration remain
classified for later research, SQLAlchemy isolation, or integration validation.
No integration or slow tests were executed in this stage.

## Research Relocations

Vision research modules were moved out of the production vision package during
the closure gate:

- `research/visual/visual_corruptions.py`
- `research/visual/visual_dataset.py`
- `research/visual/visual_encoder.py`
- `research/visual/visual_precomputed.py`
- `research/visual/visual_registration.py`
- `research/visual/visual_retrieval.py`

Video package source was not relocated to SQLAlchemy or research in this stage.

## Remaining Defects

No package-blocking defects remain for independent video wheel validation.
Several legacy mixed Stage 8 video modules remain large and are tracked by
package-path legacy quality exceptions without raising global thresholds.

The next structural stage is SQLAlchemy isolation.
