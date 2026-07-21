# Video Action-Set Finalization Integration Validation

## Purpose

This package validates the synthetic finalization path across runtime, facade,
engine, `FinalAccessService`, a file-backed SQLite store, filesystem publication,
fresh-process reconstruction, operator CLIs, PowerShell wrappers, and an installed
wheel. It is an infrastructure validation only. It does not validate scientific
provider behavior and does not authorize final access.

All databases, historical-authority files, authorizations, observations,
artifacts, receipts, virtual environments, and logs are synthetic and disposable.
The runner refuses `C:\Projects\zeromodel-stage8` and writes only beneath a unique
system temporary directory.

## Inventory Method

The inventory was collected without executing tests:

```powershell
pytest --collect-only -q --run-integration --run-slow -m "integration or slow"
```

Collection found 216 marked nodes and deselected 719 ordinary fast nodes. Of the
marked nodes, 77 are the new bounded finalization tests. The operator selection
also includes 73 existing unmarked synthetic contract nodes selected by exact
path or node ID. No marker alone was treated as evidence that a test was safe.

### Bounded Selection

| Test path | Test name or pattern | Marker | Resources | Expected | Property | Authority | Safe |
|---|---|---|---|---:|---|---|---:|
| `tests/integration/test_video_finalization_schema_authority.py` | all 10 nodes | integration | temporary SQLite files | 8 s | dedicated schema, marker/version/table/column rejection, FKs, authority separation | synthetic | yes |
| `tests/integration/test_video_finalization_sqlite_concurrency.py` | all 5 nodes | integration | two engines/connections, two bounded threads | 15 s | one CAS winner, atomic event/state update, no orphan events | synthetic | yes |
| `tests/integration/test_video_finalization_authorized_observations.py` | all 9 nodes | integration | temporary SQLite and tiny observation blob | 10 s | final-access ownership/state and non-final identity | synthetic | yes |
| `tests/test_video_final_historical_authority.py` | all 11 nodes | fast, exact selection | small temporary files/manifests | 8 s | recomputed authority, mutation timing, symlink and preflight behavior | synthetic | yes |
| `tests/integration/test_video_finalization_historical_evaluator.py` | `test_historical_manifest_bindings_*`, `test_historical_authority_rejects_relative_paths` | integration | small temporary files | 2 s | authority ID, commit, and absolute-path binding | synthetic | yes |
| `tests/test_video_final_access_kernel.py` | six evaluator test functions (15 nodes) | fast, exact selection | in-memory DTOs | 5 s | pass/fail/indeterminate, aggregates, equality, permutations, Decimal isolation, invalid rules | synthetic protocol | yes |
| `tests/integration/test_video_finalization_historical_evaluator.py` | five evaluator test functions (5 nodes) | integration | in-memory DTOs | 2 s | duplicate identity, provider set, unknown keys, incomplete evidence | synthetic protocol | yes |
| `tests/integration/test_video_finalization_executor_publication.py` | all 13 nodes | integration | temporary SQLite, filesystem, one subprocess | 12 s | hostile executor rows, service-owned decision/digests, staging mutation/symlinks, atomic promotion, registration isolation | synthetic | yes |
| `tests/test_video_final_publication.py` | all 47 nodes | fast, exact selection | temporary filesystem/in-memory store | 8 s | artifact-name matrix, staging tamper, file sets, receipt states, no merge | synthetic | yes |
| `tests/integration/test_video_finalization_failure_injection.py` | all 14 nodes | integration | temporary filesystem/in-memory store | 20 s | state/files/receipt status at every supported boundary, claim/report blocking, no repair, and no retry/rerun | synthetic | yes |
| `tests/integration/test_video_finalization_reconstruction.py` | all 7 nodes | integration | temporary SQLite/files, fresh Python processes | 35 s | durable reconstruction, terminal statuses, counters, bindings, claim eligibility | synthetic | yes |
| `tests/integration/test_video_finalization_cli_scripts.py` | all 10 nodes | integration | temporary SQLite/files, Python and PowerShell processes | 45 s | read-only commands, explicit paths, JSON/stderr, hostile IDs, exit propagation | synthetic | yes |
| `tests/integration/test_video_finalization_package_boundary.py` | installed-wheel boundary | integration | temporary source copy, wheel, virtual environment | 90 s | packaged modules, no executor, no historical `--build-final`, early final-build prohibition | synthetic | yes |

### Excluded Marked Inventory

| Test path | Test name or pattern | Marker | Resources | Expected | Property | Authority | Safe |
|---|---|---|---|---:|---|---|---:|
| `tests/integration/test_video_provider_measurement_real.py` | real P1/P2/P3 golden | integration | repository identity, reachability tile, real provider scoring | variable | real provider evidence golden | real/preserved | **no** |
| `tests/test_video_action_set_benchmark.py` | all 9 nodes | auto integration | complete benchmark plans/materialization | minutes | benchmark generation and final freeze | scientific | **no** |
| `tests/test_video_action_set_claim_quarantine.py` | claim quarantine | auto integration | repository result fixtures | variable | claim status files | preserved | **no** |
| `tests/test_video_action_set_family_reachability.py` | all 9 nodes | auto integration | family materialization and transformations | minutes | scientific family reachability | scientific | **no** |
| `tests/test_video_action_set_family_semantics.py` | 18 nodes | auto integration; one slow | complete semantic universes/replay | minutes | scientific family semantics and closure | scientific | **no** |
| `tests/test_video_action_set_instrument.py` | both nodes | auto integration | audits and mutation gate | minutes | instrument/mutation behavior | scientific | **no** |
| `tests/test_video_action_set_reference_verification.py` | 16 nodes | auto integration; two slow | reference fixtures and complete mutation audit | minutes | full reference and mutation closure | preserved/scientific | **no** |
| `tests/test_arcade_visual_local_baseline_showdown.py` | artifact generation | integration + slow | model/image generation outputs | minutes | visual showdown | scientific | **no** |
| `tests/test_arcade_visual_registered_calibration_v2.py` | artifact generation | integration + slow | calibration output | minutes | registered calibration | scientific | **no** |
| `tests/test_installed_wheel_video_instrument.py` | both nodes | integration + slow | broad wheel build/install | minutes | whole-instrument packaging | synthetic but broad | **no** |
| `tests/test_video_discriminative_evidence.py` | all 11 nodes | integration; one slow | discriminative masks/regions | variable | discriminative evidence | scientific | **no** |
| `tests/test_video_discriminative_measurement_audit.py` | all 7 nodes | integration + slow | preserved pre-final artifacts | minutes | measurement audit | preserved/scientific | **no** |
| `tests/test_video_discriminative_representation_audit.py` | all 3 nodes | integration + slow | representation audit fixtures | minutes | representation audit | preserved/scientific | **no** |
| `tests/test_video_discriminative_v2_benchmark.py` | all 3 nodes | integration + slow | full universe and output writes | minutes | V2 benchmark freeze | scientific | **no** |
| `tests/test_video_discriminative_v2_integrity.py` | all 4 nodes | integration + slow | full universe verification | minutes | V2 integrity | scientific | **no** |
| `tests/test_video_discriminative_v2_selection.py` | all 3 nodes | integration + slow | selection/calibration rebuild | minutes | V2 selection | scientific | **no** |
| `tests/test_video_episode_plan_sql_store.py` | all 9 nodes | integration | in-memory SQLite | seconds | generic plan persistence | synthetic | not selected; replaced by focused finalization coverage |
| `tests/test_video_identity_rmdto.py` | core import boundary | integration | import graph | <1 s | SQLAlchemy boundary | synthetic | not selected; unrelated |
| `tests/test_video_identity_sql_store.py` | all 5 nodes | integration | in-memory SQLite | seconds | generic identity persistence | synthetic | not selected; unrelated |
| `tests/test_video_local_correlation.py` | all 7 nodes | integration + slow | generated benchmark/provider | minutes | local-correlation science | scientific | **no** |
| `tests/test_video_observation_sql_store.py` | all 14 nodes | integration | in-memory SQLite and arrays | seconds | generic observation persistence | synthetic | not selected; focused final ownership test used |
| `tests/test_video_prospective_providers.py` | all 3 nodes | integration + slow | provider scoring over 112 rows | minutes | prospective providers | scientific | **no** |
| `tests/test_video_prospective_runtime_equivalence.py` | runtime equivalence | integration + slow | provider profiles | minutes | provider optimization equivalence | scientific | **no** |
| `tests/test_visual_local_baseline_result_records.py` | result reconstruction | integration + slow | preserved result bundle | variable | result records | preserved | **no** |
| `tests/test_visual_local_baselines.py` | both nodes | integration + slow | registered pixel providers | variable | local baseline science | scientific | **no** |
| `tests/test_visual_registered_calibration_v2.py` | all 3 nodes | integration + slow | calibration grids | minutes | registered pixel calibration | scientific | **no** |
| `tests/test_visual_result_records.py` | both nodes | integration + slow | result manifests/atlas | variable | visual result records | preserved | **no** |

No selected test uses network access, a real project path, preserved Stage 8
evidence, a repository scientific fixture, or a production executor. Tests that
exercise symlink behavior skip only when the operating system denies symlink
creation.

## Prerequisites

- A clean checkout on `codex/video-finalization-integration-validation`.
- The full 40-character commit SHA intended for validation.
- Python with the repository development dependencies already installed.
- Windows PowerShell for the two wrapper smoke cases. Those cases skip if it is
  unavailable.
- No network is required. The wheel group uses `--no-isolation`, `--no-index`,
  and `--no-deps` from a temporary source copy.

## Operator Command

Determine and review the exact commit, then pass it explicitly:

```powershell
$commit = git -C 'C:\Projects\zeromodel-finalization' rev-parse HEAD
powershell -NoProfile -File 'C:\Projects\zeromodel-finalization\scripts\run-video-finalization-integration.ps1' `
    -RepositoryPath 'C:\Projects\zeromodel-finalization' `
    -ExpectedCommit $commit
```

The runner rejects a short SHA, wrong branch, wrong commit, dirty worktree,
forbidden Stage 8 path, or failed group. It performs no retry. To delete fixtures
for successful groups, add `-RemoveSuccessfulFixtures`; failed fixtures are always
preserved.

## Groups And Budget

| Group | Predicted | Hard timeout |
|---|---:|---:|
| `01-schema-authority` | 8 s | 60 s |
| `02-sqlite-transactions` | 15 s | 90 s |
| `03-authorized-observations` | 10 s | 60 s |
| `04-historical-authority` | 10 s | 60 s |
| `05-evaluator` | 8 s | 60 s |
| `06-publication` | 20 s | 90 s |
| `07-failure-injection` | 20 s | 90 s |
| `08-reconstruction` | 35 s | 120 s |
| `09-cli-scripts` | 45 s | 120 s |
| `10-package-boundary` | 90 s | 180 s |

Predicted total is about 4.5 minutes. The target is under 5 minutes and the hard
operator budget is under 10 minutes. A group timeout is recorded as exit code 124.

## Outputs

The runner prints its unique validation root. Its model is:

```text
%TEMP%\zeromodel-finalization-validation-<guid>\
  .zeromodel-synthetic-validation
  run-state.json
  summary.json
  summary.md
  logs\<group>.stdout.log
  logs\<group>.stderr.log
  fixtures\<group>\
```

All logs and fixtures are outside the repository, canonical final artifact paths,
and scientific output directories. Inspect a running or completed validation once
without polling scientific state:

```powershell
powershell -NoProfile -File 'C:\Projects\zeromodel-finalization\scripts\observe-video-finalization-integration.ps1' `
    -ValidationRoot '<root printed by the runner>'
```

The observer reads only the synthetic marker, run state, summary presence, log
metadata, and runner/group process state. It writes nothing.

## Interpreting Results

Success requires all ten groups to report `passed`, each exit code to be zero,
`summary.json` to report `synthetic_only: true`, and both summary files to exist.
On failure, inspect the named stdout/stderr logs and preserved fixture directory.
Do not rerun automatically; determine whether the failure is deterministic first.

Passing proves the bounded infrastructure contracts listed in the selection
matrix on the tested operating system and package environment. It does not prove
real provider formulas, real P1/P2/P3 measurements, real final reachability,
scientific correctness, mutation closure, networked or cross-machine exactly-once
behavior, or compatibility with preserved Stage 8 evidence.

Running this package does not create an approved real protocol, create a live
authorization, register a production executor, reserve real final access,
materialize or score final observations, publish a real receipt, or authorize any
subsequent final action.
