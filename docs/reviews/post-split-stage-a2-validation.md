# Post-split remediation — Stage A2: verification completeness and contract restoration

**Baseline SHA (start of this stage):** `b8f75b85f298f974b850df2da7f72b89cd4f12d9` (branch `main`, working tree clean at start — confirmed via `git branch --show-current`, `git rev-parse HEAD`, `git status --short` before any change).
**Final SHA / working-tree state:** uncommitted at the time of writing — every change described below is staged in the working tree, pending an explicit commit decision (nothing was committed, tagged, published, or versioned in this stage).
**Objective:** make every production verification signal (`run_fast_tests.py`, `check_quality.py`, the public API manifest, identity evidence) complete and truthful, without redesigning the package system. Findings addressed: F4, F5, F6, F7, H2, H3, H4, M3, M6, M7, M9, L4 (from [post-split-main-audit.md](post-split-main-audit.md)).

## Files changed (by area)

- **Preflight (§1):** `pyproject.toml` (deduplicated + retitled the 5-marker taxonomy), `tests/conftest.py` (removed duplicate marker registration; added `integration_tests` path recognition and `research` path/prefix/file auto-tagging), `research/video_action_set/benchmarks/arcade_visual_video_local_correlation_benchmark.py` (`parents[1]` → `parents[3]`, module-level `# ruff: noqa: E402`).
- **Fast suite (§2-3):** `scripts/run_fast_tests.py` (rewritten: explicit six-package + two repository-wide test roots, `not slow and not external and not research` marker expression, `--run-integration` to defeat the legacy blanket-integration-deselection, structured JSON reporting), `scripts/fast_suite_reporter.py` (new pytest plugin providing collected/deselected/passed/failed/skipped/collection-error counts without parsing terminal text).
- **SQLAlchemy quality gate (§4-5):** `scripts/check_quality.py` (added `packages/sqlalchemy/src`, `packages/sqlalchemy/tests`, `integration_tests` to every governed path list; reordered stages to format→lint→mypy→boundaries→architecture→limits; each stage now prints its paths), `pyproject.toml` (`mypy_path` += `packages/sqlalchemy/src`), `packages/sqlalchemy/src/.../db/stores/video_action_set.py` (fixed the one real mypy defect: `isinstance(result, CursorResult)` narrowing instead of assuming `.rowcount` exists), `packages/sqlalchemy/tests/test_sqlalchemy_package_isolation.py` (6 test-only typing fixes: precise `_sql_store` return type, `isinstance` narrowing instead of `hasattr`, a `dict` narrowing assert), `packages/sqlalchemy/src/.../video_action_set_final_cli.py` (pre-existing formatting drift, now caught by the gate), `quality-baseline.toml` (one new legacy-size exception for `db/stores/video_action_set.py`, following the exact pattern already used for ~20 other pre-split-legacy files).
- **Public API manifest (§6):** `scripts/validate_release_candidate.py` (`write_public_exports()` rewritten to introspect real `__all__` in the clean venv; `PACKAGES` gained `depends_on` edges).
- **Identity (§7):** `docs/architecture/package-core-validation-1.0.13.md` (corrected golden digest + full fixture record), `docs/reviews/post-split-identity-consistency.md` (resolution appended).
- **Claims-audit (§8):** `.github/workflows/claims-audit.yml` (path check now covers `packages/`, `tests/`, `integration_tests/`, `scripts/`, `README.md`, `package-boundaries.toml` instead of the deleted `zeromodel/`).
- **Integration marker (§9):** covered by the `tests/conftest.py` change above (`integration_tests` in `item.path.parts`).
- **Test ownership (§10):** `research/video_action_set/tests/test_video_policy.py` (trimmed from 11 to 1 test), new `packages/video/tests/test_video_policy_reader_contracts.py` (the other 10, with a synthetic fixture), `research/video_action_set/tests/test_video_benchmark_facade.py` and `test_video_verification_closure_kernel.py` (`parents[1]`→`parents[3]` fix, explicit `research` marker), 4 obsolete `tests/integration/*.py` files repaired (stale flat module-path strings → current `zeromodel.persistence.sqlalchemy.*` / `zeromodel.video.domains.*` paths; one file - `test_video_finalization_package_boundary.py` - rewritten outright since its premise built the retired monolithic root wheel), `scripts/video-final-*.ps1` (same stale-path fix, 4 files), `tests/test_installed_wheel_video_instrument.py` (rewritten: was vacuously checking a nonexistent root `zeromodel/` tree and asserting research-only modules *should* be importable from a production wheel - now checks all six real source roots and the correct, opposite invariant).
- **Release-validator reporting (§11):** `scripts/validate_release_candidate.py` (`install_and_probe()` now returns per-package wheel-smoke results; new `release_test_layer_report()` writes `docs/architecture/package-release-test-layers-1.0.13.json`).
- **Developer tooling (§12):** `.vscode/settings.json` (trailing newline added; contents already matched the canonical six-package test roots after an earlier commit).
- **Quality-gate ordering (§13):** covered by the `check_quality.py` rewrite above.
- **Docs:** this file, plus the other three §17 deliverables, plus a remediation-status section appended to `post-split-main-audit.md`.

## Production test roots included in the canonical fast suite

```
tests
integration_tests
packages/core/tests
packages/analysis/tests
packages/observation/tests
packages/vision/tests
packages/video/tests
packages/sqlalchemy/tests
```

Marker expression: `not slow and not external and not research` (integration tests are included by default now; only slow/external/research are excluded by marker, never by directory omission).

## Fast-suite counts (final run, this session)

```
Collected: 874, deselected: 79, passed: 794, failed: 0, skipped: 1
Fast-suite runtime: 100.06s (budget: 120s)
```

Structured JSON report: `build/reports/fast-test-summary.json` (git-ignored build artifact, regenerated every run).

**Margin note:** 100s of a 120s budget (~83%) on this development machine. This is real headroom, not a hidden failure, but it is tighter than Stage A1's 40s baseline - the six package-local suites plus several wheel-building integration tests (`tests/integration/test_video_finalization_*`, `tests/test_installed_wheel_video_instrument.py`) now legitimately run inside the fast suite where they previously did not. If CI hardware is slower than this machine, the budget may need raising; flagged here rather than silently risking future flakiness.

## Package-local counts by package

| package | passed | skipped |
|---|---|---|
| core | 20 | 1 (Lua interpreter not installed locally) |
| analysis | 65 | 0 |
| observation | 32 | 0 |
| vision | 10 | 0 |
| video | 20 (10 pre-existing + 10 new from the test_video_policy.py split) | 0 |
| sqlalchemy | 6 | 0 |

## Integration count

108 passed (0 failed) when explicitly selected via `--run-integration -m integration` across `tests`, `integration_tests`, and all six `packages/*/tests` - up from a smaller number before this stage's obsolete-test repairs (4 previously-broken `tests/integration/*.py` files now pass instead of erroring).

## Research collection status

`research/video_action_set/tests/` (30 files): the 4 named files were addressed per their individual classification (see [post-split-stage-a2-test-ownership-changes.csv](post-split-stage-a2-test-ownership-changes.csv)). Overall collection improved from Stage A1's 25 collected/21 errors to **36 collected**, with 2 files (`test_video_benchmark_facade.py`, `test_video_verification_closure_kernel.py`) each having 2 of their tests fail on pre-existing stale `architecture_rules` module-name constants (documented, not fixed - out of this stage's scope per the audit's own note that fully migrating those ~40 constants requires domain knowledge not safe to fabricate). `test_video_local_correlation.py` was confirmed genuinely expensive (exceeded a 2-minute timeout in this session), consistent with its research classification. Research failures were not treated as production failures anywhere in this stage; `docs/reviews/post-split-research-health.md` (Stage A1) remains the authoritative research-health record and was not rewritten.

## SQLAlchemy

- Ruff format/lint: passing (one pre-existing formatting drift in `video_action_set_final_cli.py` fixed, now that the gate actually inspects it).
- mypy: **0 errors** (was 17 before Stage A1 added `sqlalchemy` to `mypy_path`; was 7 after that fix surfaced the real, previously-masked issues; now 0 after fixing the one production defect and 6 test-only typing issues - see [post-split-stage-a2-validation.md](post-split-stage-a2-validation.md) file-change list above for exactly what changed).
- `python -m mypy packages/sqlalchemy/src packages/sqlalchemy/tests` passes with **no package-wide suppression** - every fix is a precise, local, runtime-justified narrowing (`isinstance` checks) or a corrected type annotation, never a blanket ignore.

## Architecture module count

112 production modules (`scripts/check_architecture.py` and `scripts/check_package_boundaries.py` agree exactly), unchanged from Stage A1 - this stage did not touch module discovery, only quality-gate coverage and test classification.

## Public API symbol count

**211** real per-symbol rows across all six distributions (core 52, analysis 87, observation 12, vision 18, video 32, sqlalchemy 10) - up from 6 placeholder rows. See [post-split-stage-a2-api-manifest-validation.md](post-split-stage-a2-api-manifest-validation.md) for full verification detail.

## Identity resolution

Core golden artifact-kernel digest: `32f801671139b73e349c756570c27c06d39c422a4d9a277782e1c997a473083b` (the value the enforced test, `packages/core/tests/test_artifact_kernel.py`, has always pinned). `package-core-validation-1.0.13.md`'s previously-incorrect value was corrected to match; `package-integration-validation-1.0.13.md` already had the correct value. See [post-split-identity-consistency.md](post-split-identity-consistency.md) for the full resolution record, and [post-split-stage-a2-api-manifest-validation.md](post-split-stage-a2-api-manifest-validation.md) is unrelated (kept separate per deliverable naming) - identity detail lives in the identity-consistency doc.

## Claims-workflow corrections

`.github/workflows/claims-audit.yml`'s `package_changed` check now reacts to `packages/**`, `tests/**`, `integration_tests/**`, `scripts/**`, `README.md`, and `package-boundaries.toml`, replacing the dead `zeromodel/**` check that could never match anything in the current layout. Verified with `tests/test_claims_audit_workflow_coverage.py` (3 tests) and a repo-wide scan (`tests/test_workspace_ci_invariants.py`'s existing invariant plus a new one) confirming no active workflow trigger path still references the deleted root tree.

## Ruff / quality-gate / release-validator results (this session)

```
python scripts/check_package_boundaries.py   -> Package boundary check passed: 112 production modules
python scripts/check_architecture.py         -> Architecture check: passed (112 production modules inspected)
python scripts/check_quality.py              -> Quality checks passed (all 6 stages, all 6 packages, in the required order)
python -m mypy packages/sqlalchemy/src packages/sqlalchemy/tests -> Success: no issues found in 13 source files
python scripts/validate_release_candidate.py -> Release candidate validation passed
```

`ruff format --check .` / `ruff check .` (whole repository, not just the governed gate): the governed gate (`check_quality.py`'s paths) is 100% clean. The unscoped whole-repo run additionally reports 141 files with pre-existing formatting drift and 2 pre-existing `E402` lint findings, all outside `check_quality.py`'s governed paths (in `tests/`, `research/`, `examples/` files never touched by this stage or its predecessor). These are not new regressions - reformatting 141 unrelated files was judged out of scope for a stage about verification-signal truthfulness, and is noted here rather than silently mass-reformatted or silently omitted.

## Remaining findings for a later stage

- The 2 research test failures from stale `architecture_rules` module-name constants (documented, not fixed).
- `test_video_benchmark_facade.py` / `test_video_verification_closure_kernel.py`'s recommended merge-and-relocate (both test `scripts/check_architecture.py`'s layering rules with overlapping edges; consolidating them was recommended in Stage A1 and re-confirmed here, but not executed - not necessary for production-contract truthfulness).
- The 141-file whole-repository ruff-format drift outside the governed gate.
- Full research-suite health beyond the 4 named files (see `post-split-research-health.md`).
- Any further Linux-CI-only verification of the per-package Windows-only workflows (`analysis-package.yml` etc.) - unchanged in this stage, out of its scope.

## Confirmation

Nothing was published, tagged, or version-bumped in this stage. No commit was made or push performed - all changes described above remain in the working tree pending your explicit go-ahead.
