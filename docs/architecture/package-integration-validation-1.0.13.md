# ZeroModel 1.0.13 Final Package Integration Validation

Date: 2026-07-22

## Verdict

Release-candidate validation passed locally for the six-distribution namespace
split. No TestPyPI upload, PyPI upload, tag, or GitHub release was performed.

## Package Set

| Distribution | Version | Namespace | Direct dependencies |
|---|---:|---|---|
| `zeromodel` | 1.0.13 | `zeromodel.core` | `numpy>=1.23` |
| `zeromodel-analysis` | 1.0.13 | `zeromodel.analysis` | `numpy>=1.23`, `zeromodel==1.0.13` |
| `zeromodel-observation` | 1.0.13 | `zeromodel.observation` | `numpy>=1.23`, `zeromodel==1.0.13` |
| `zeromodel-vision` | 1.0.13 | `zeromodel.vision` | `numpy>=1.23`, `zeromodel==1.0.13`, `zeromodel-observation==1.0.13` |
| `zeromodel-video` | 1.0.13 | `zeromodel.video` | `numpy>=1.23`, `zeromodel==1.0.13`, `zeromodel-observation==1.0.13` |
| `zeromodel-sqlalchemy` | 1.0.13 | `zeromodel.persistence.sqlalchemy` | `numpy>=1.23`, `SQLAlchemy>=2.0,<3`, `zeromodel==1.0.13`, `zeromodel-video==1.0.13` |

## Provider Measurement Blocker

`tests/integration/test_video_provider_measurement_real.py` was stale
research-owned benchmark coverage. It was moved to
`research/video_action_set/tests/test_video_provider_measurement_real.py` and
now imports `research.video_action_set.provider_measurement`. The production
`provider_measurement` module was not restored.

## Validation Commands

| Command | Result |
|---|---|
| `python scripts/validate_release_candidate.py` | Passed; built and checked all six wheels and sdists, installed all wheels into `build/full-integration-venv`, ran `pip check`, verified namespace imports, rejected root `from zeromodel import ScoreTable`, and wrote the artifact/API manifests. |
| `python scripts/run_fast_tests.py` | Passed; `598 passed, 215 deselected` in 75.41 seconds. |
| `python scripts/check_quality.py` | Passed; formatting, linting, typing, architecture, and quality limits. |
| `python -m pytest integration_tests/test_package_integration_smoke.py -q` | Passed; one bounded integration smoke across core, analysis, observation, vision, video, memory store, and SQLAlchemy store. |
| `python -m build` from repository root | Failed as expected for the old flat-layout root project; 1.0.13 builds from `packages/*` through `scripts/validate_release_candidate.py`. |
| `python -m ruff check .` | Failed on pre-existing example/research cleanup and generated-inventory style issues outside the scoped release quality gate. |

## Artifact Manifest

The authoritative artifact list with SHA-256 digests is
`docs/architecture/package-release-artifacts-1.0.13.json`.

| Distribution | Wheel SHA-256 prefix | Sdist SHA-256 prefix |
|---|---|---|
| `zeromodel` | `3caac617a704` | `7dcfaf05abe7` |
| `zeromodel-analysis` | `1bc85512ba0d` | `cc662e72a8b3` |
| `zeromodel-observation` | `6cdca297dab4` | `331d6eb73594` |
| `zeromodel-vision` | `171a49bdadbb` | `b8310221bfcc` |
| `zeromodel-video` | `b82a2392348a` | `492885074108` |
| `zeromodel-sqlalchemy` | `70b7f075c9aa` | `07af319dce6d` |

## Identity Regression

The core package golden remains
`32f801671139b73e349c756570c27c06d39c422a4d9a277782e1c997a473083b` for the
core artifact kernel fixture. The cross-package analysis smoke uses
`key=value` row IDs and therefore has a separate deterministic artifact identity:
`3ce8dd265b949b3b26ebcd602c8b572c248b25c5bafdd13b459a1ab739533e4a`.

## Release Boundary

The release workflow for this stage stops at release-candidate validation. The
repository now includes `.github/workflows/package-integration.yml` to run the
same six-package validation across Python 3.10, 3.11, and 3.12. Publishing,
tagging, and GitHub release creation remain operator actions after review.
