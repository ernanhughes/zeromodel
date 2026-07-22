# ZeroModel post-split repository audit

**Baseline:** `c1ce710db50655a6082567fd3f376c3134095ea2` (confirmed identical to the working-tree `HEAD` at audit time; working tree clean).
**Scope:** repository-wide development, test, CI, quality, and release machinery for the six-distribution architecture (`zeromodel`, `zeromodel-analysis`, `zeromodel-observation`, `zeromodel-vision`, `zeromodel-video`, `zeromodel-sqlalchemy`, all at version `1.0.13`).
**Method:** evidence-only. No production code, tests, workflows, or packaging metadata were changed. All claims below were independently reproduced against the baseline SHA, mostly from a disposable `git worktree` checked out at that exact commit plus a disposable venv built via `pip install -r requirements-dev.txt`, so results are not contaminated by this machine's local, untracked build artifacts (a stray root `zeromodel/` directory, `zeromodel.egg-info/`, `build/`, `dist/`, `venv/` all exist locally but are confirmed absent from git and correctly `.gitignore`d).

Companion deliverables (read alongside this document, not duplicated in full here):
- [post-split-ci-command-matrix.csv](post-split-ci-command-matrix.csv) - every executed command in every workflow, with liveness verdicts
- [post-split-release-validator-matrix.csv](post-split-release-validator-matrix.csv) - feature-by-feature verification of `scripts/validate_release_candidate.py`
- [post-split-quality-coverage.csv](post-split-quality-coverage.csv) - per-package-root coverage by ruff/mypy/quality-limits/architecture
- [post-split-test-ownership.csv](post-split-test-ownership.csv) - all 140 test files classified by asserted behavior
- [post-split-research-health.md](post-split-research-health.md) - research collection results and split recommendations
- [post-split-identity-consistency.md](post-split-identity-consistency.md) - golden-identity and public-API-manifest cross-checks

## Ranked findings

### BLOCKER

---
**F1 - Root `pyproject.toml` has no `[project]` table; every CI command that installs `.[dev]`/`-e .[dev]`/`-e .[release]`/`-e '.[vision]'` fails immediately**
- **Affected files:** `pyproject.toml`; `.github/workflows/python.yml` (all 4 jobs: `quality`, `package-tests`, `lua-edge`, `package-build`); `.github/workflows/integration.yml`; `.github/workflows/publish-testpypi.yml`; `.github/workflows/visual-address-benchmark.yml`
- **Exact evidence:** `python -m pip install -e ".[dev]"` at repo root reproduces: `error: Multiple top-level packages discovered in a flat-layout: ['site', 'packages', 'research', 'zeromodel', 'integration_tests']` (setuptools auto-discovery, since root has no `[project]`/`[build-system]` table to disambiguate). This is not wrapped in `continue-on-error` in any of the affected jobs, so each job hard-fails at the install step, before ever reaching the command it exists to run.
- **Why it matters:** `python.yml` is the workflow named "Python package" that `docs/release.md`'s own "Review and merge" section requires to pass before merging a release PR - it currently cannot ever pass. `publish-testpypi.yml` and `integration.yml` are equally dead.
- **Recommended owner:** CI/release maintainer
- **Recommended implementation stage:** Immediate (blocks release process and gives a permanently red required check)
- **Acceptance test:** Each affected workflow either installs via `pip install -r requirements-dev.txt` (or equivalent per-package `-e packages/<name>`) instead of root extras, or is retired/replaced by its six-package equivalent; a workflow run on a fresh checkout completes past the install step.

---
**F2 - `scripts/validate_release_candidate.py` hardcodes a Windows venv path; the only Linux-wide six-package validation always fails**
- **Affected files:** `scripts/validate_release_candidate.py:214`; `.github/workflows/package-integration.yml`
- **Exact evidence:** `install_and_probe()` sets `python = venv / "Scripts" / "python.exe"` unconditionally - no `sys.platform`/`os.name` branching anywhere in the file. `package-integration.yml` runs this script with `runs-on: ubuntu-latest` across a 3.10/3.11/3.12 matrix; a POSIX venv has `bin/python`, not `Scripts/python.exe`, so `subprocess.run` raises a file-not-found error on every invocation.
- **Why it matters:** This is the one workflow meant to validate the full six-package release candidate together on Linux. It has never successfully completed the "Release-candidate package validation" step since being wired to `ubuntu-latest`.
- **Recommended owner:** release-tooling maintainer
- **Recommended implementation stage:** Immediate
- **Acceptance test:** `install_and_probe()` selects `bin/python` on POSIX and `Scripts/python.exe` on Windows (e.g. via `"Scripts" if os.name == "nt" else "bin"`); `package-integration.yml` completes all three matrix legs.

---
**F3 - `scripts/check_architecture.py` still points at the deleted root distribution; the "Architecture" quality gate silently checks zero files**
- **Affected files:** `scripts/check_architecture.py:10`; `scripts/check_quality.py` (Architecture step); `.github/workflows/python.yml`; `.github/workflows/package-integration.yml`
- **Exact evidence:** `PACKAGE_ROOT = REPO_ROOT / "zeromodel"`. Verified in a clean worktree at the baseline SHA: root `zeromodel/` does not exist; `discover_modules()` returns `{}`; the script prints `Architecture check: passed` having examined 0 modules. This is a different script from `scripts/check_package_boundaries.py` (which correctly reads `package-boundaries.toml`'s `packages/*/src` roots and does work - independently verified: `Package boundary check passed: 112 production modules`), but `check_quality.py` calls the broken one, not the working one.
- **Why it matters:** Every green "Architecture: passed" line in `check_quality.py` output (used by `python.yml` and `package-integration.yml`) has been meaningless since the split - it provides zero import-cycle or forbidden-edge protection for any of the six packages.
- **Recommended owner:** quality-tooling maintainer
- **Recommended implementation stage:** Immediate
- **Acceptance test:** `check_architecture.py` is repointed at `packages/*/src` (or retired in favor of `check_package_boundaries.py`, which already covers this ground correctly); rerunning against a deliberately introduced cross-package import cycle fails the check.

---
**F4 - `zeromodel-sqlalchemy` has zero ruff/mypy coverage in the repository-wide quality gate, and independently already fails mypy**
- **Affected files:** `scripts/check_quality.py:8-34` (`FORMAT_LINT_PATHS`, `TYPING_PATHS`); root `pyproject.toml:25` (`[tool.mypy] mypy_path`)
- **Exact evidence:** `packages/sqlalchemy` appears in neither list. Running `mypy packages/sqlalchemy/src` directly (outside the gate) produces 17 errors across 9 files: missing `py.typed` marker on every internal cross-module import, plus one genuine bug at `db/stores/video_action_set.py:1070` (`Result[Any]` has no attribute `rowcount`). `ruff check`/`ruff format --check` on the same tree pass cleanly, so only the mypy gap is currently masking a real defect.
- **Why it matters:** The sixth distribution is invisible to the gate that every PR touching the other five packages must pass; a regression in sqlalchemy's typing would never fail CI via this path (it is separately, redundantly caught by `sqlalchemy-package.yml`'s own ad hoc mypy step, but that workflow is Windows-only - see F-H3).
- **Recommended owner:** quality-tooling maintainer
- **Recommended implementation stage:** Immediate
- **Acceptance test:** `packages/sqlalchemy/src` and `packages/sqlalchemy/tests` added to both lists and to `mypy_path`; `check_quality.py` fails on the existing `rowcount` bug until it is fixed.

---
**F5 - The "bounded fast suite" never collects any of the 144 package-local tests**
- **Affected files:** `pyproject.toml:4` (`testpaths = ["tests", "integration_tests"]`); `scripts/run_fast_tests.py`
- **Exact evidence:** `run_fast_tests.py` runs bare `pytest -q --maxfail=1` with no path arguments, relying entirely on `testpaths`. Reproduced: `pytest -q --collect-only` from repo root collects from `tests/`+`integration_tests/` only (449 selected / 186 deselected of 635 total with default markers); `pytest -q --collect-only packages/core/tests packages/analysis/tests packages/observation/tests packages/vision/tests packages/video/tests packages/sqlalchemy/tests` separately collects **144** tests untouched by the first command.
- **Why it matters:** The one command CI calls "the complete bounded fast suite" (`python.yml`'s `package-tests` job, `package-integration.yml`'s "Fast suite" step) validates zero package-local unit tests for any of the six distributions. Package-local coverage only happens via the five per-package workflows building and testing installed wheels, which is a different (and, per F-H3, Windows-only for five of six) code path.
- **Recommended owner:** test-infrastructure maintainer
- **Recommended implementation stage:** Immediate
- **Acceptance test:** `testpaths` includes `packages/*/tests` (or `run_fast_tests.py` passes explicit paths), and a deliberately broken package-local test fails `python scripts/run_fast_tests.py` from repo root.

---
**F6 - `docs/architecture/package-public-api-1.0.13.csv` is not a real public-API manifest**
- **Affected files:** `scripts/validate_release_candidate.py:311-337` (`write_public_exports`); `docs/architecture/package-public-api-1.0.13.csv`
- **Exact evidence:** The function writes exactly one row per package with the literal string `"__all__"` hardcoded into the `exported_symbol` column - never an actual symbol name. The live CSV has 6 data rows. Independently counting every package's real `__all__` gives 211 total public symbols (core 52, analysis 87, observation 12, vision 18, video 32, sqlalchemy 10). The CSV captures roughly 2.8% of the real surface (6 marker rows instead of 211 per-symbol rows); its `namespace`/`owning_module` columns are correct but are hardcoded constants, not introspected.
- **Why it matters:** Anything (tooling or a reviewer) that consults this file to answer "is symbol X part of package Y's public API" gets no real answer today. See [post-split-identity-consistency.md](post-split-identity-consistency.md) for the full per-package table.
- **Recommended owner:** release-tooling maintainer
- **Recommended implementation stage:** Immediate
- **Acceptance test:** `write_public_exports()` introspects each package's actual `__all__` (or public-symbol AST scan) and writes one row per symbol; regenerated CSV has 211 rows (or the then-current true count) matching a fresh scan.

---
**F7 - `package-core-validation-1.0.13.md`'s golden artifact-kernel identity is factually wrong and contradicted by the integration report**
- **Affected files:** `docs/architecture/package-core-validation-1.0.13.md`; `docs/architecture/package-integration-validation-1.0.13.md`; `packages/core/tests/test_artifact_kernel.py:16-18`
- **Exact evidence:** Core report states the golden ID is `32f8013789e4ff463569e2ccbbdc8c3802bc42c6edeb8ceb361afca9a6025db1`; the integration report states `32f801671139b73e349c756570c27c06d39c422a4d9a277782e1c997a473083b` for the same fixture. The actual test pins the integration report's value. `git log --all -S` on the core report's value returns exactly one hit (the markdown line itself); the correct value already existed in the pre-split root test several commits earlier - i.e. the core report's number was never correct at any point in history.
- **Why it matters:** A "golden identity" section exists specifically to let a reader trust that a documented hash is enforced by a real, reproducible test. Here it is not, and a second report flatly contradicts it - readers have no way to know which (if either) is authoritative without independently re-deriving it, which is exactly what this audit had to do.
- **Recommended owner:** documentation owner for `docs/architecture/package-core-validation-1.0.13.md`
- **Recommended implementation stage:** Immediate (documentation-only fix, no code risk)
- **Acceptance test:** Core report's golden value is corrected to match `test_artifact_kernel.py`'s `GOLDEN_SAMPLE_ARTIFACT_ID`; both reports agree.

---
**F8 - There is no working release-publish path for the six-package architecture, and `docs/release.md` contradicts the repository's actual state**
- **Affected files:** `.github/workflows/publish-testpypi.yml`; `scripts/create-release.ps1`; `docs/release.md`
- **Exact evidence:** `publish-testpypi.yml` installs `-e .[release]` and runs `python -m build` at repo root - both fail for the same reason as F1 (non-buildable root), and even if fixed it only ever builds the root distribution, never `packages/<name>`. `scripts/create-release.ps1:117-118` does `Replace-One (Join-Path $Root "pyproject.toml") '(?m)^version\s*=\s*"[^"]+"\s*$'` and `Replace-One (Join-Path $Root "zeromodel\__init__.py") ...` - root `pyproject.toml` has no `version =` line at all, and `zeromodel\__init__.py` does not exist in the repo. `docs/release.md`'s closing section states: "After ZeroModel 1.0.12 is published, begin the package-architecture work under: `2.0.0.dev0`. The package split ... belongs to the 2.x development line rather than 1.0.13" - describing the split as *future* work, while every `packages/*/pyproject.toml` already declares version `1.0.13` today.
- **Why it matters:** The only actual release automation that exists (`validate_release_candidate.py`) performs local build/check/install validation but no orchestration (no branch, PR, tag, or PyPI upload logic for the six-package model - see the "MISSING" rows in [post-split-release-validator-matrix.csv](post-split-release-validator-matrix.csv)). The only orchestration script that does exist is provably built for, and only for, the retired single-package model. If a release were attempted today via the documented process, it would fail at the first file-rewrite step.
- **Recommended owner:** release-process owner
- **Recommended implementation stage:** Before the next real release is attempted (not necessarily "immediate" in the sense of blocking today's CI, but must precede any actual publish)
- **Acceptance test:** `docs/release.md` describes one process that matches reality; either `create-release.ps1` is rewritten for the six-package model (multi-file version bump, no `zeromodel\__init__.py` reference) or is explicitly retired with a stated replacement; a dry run of the documented process completes past the version-bump step.

### HIGH

---
**H1 - `claims-audit.yml`'s governance gate is permanently dormant for the current layout**
- **Affected files:** `.github/workflows/claims-audit.yml:38`
- **Exact evidence:** `package_changed = any(path.startswith("zeromodel/") for path in changed)`. Real package source now lives under `packages/<name>/src/zeromodel/...`, never under a bare root `zeromodel/` prefix (confirmed absent from git). `package_changed` is therefore always `False` for genuine production-code changes, so no PR can ever be required to update `docs/claims-audit.md` again, regardless of what ships.
- **Why it matters:** This is a governance/compliance gate, not a build gate - its silent failure mode (never blocks, never even fails visibly) is worse than a crash, since nobody would notice.
- **Recommended owner:** governance-tooling maintainer
- **Recommended implementation stage:** Immediate
- **Acceptance test:** `package_changed` checks `path.startswith("packages/")` (or per-package prefixes); a PR touching `packages/core/src/...` without touching `docs/claims-audit.md` and without an `Audit-Exempt:` line fails the gate.

---
**H2 - Three `tests/integration/` files are fully obsolete, and the same dead module is invoked by shipped PowerShell tooling, not just tests**
- **Affected files:** `tests/integration/test_video_finalization_cli_scripts.py`, `tests/integration/test_video_finalization_package_boundary.py`, `tests/integration/test_video_finalization_reconstruction.py`, `scripts/video-final-observe.ps1`, `scripts/video-final-reconstruct.ps1`
- **Exact evidence:** Every test in these three files subprocesses into or imports flat pre-split module paths - `zeromodel.video_action_set_final_cli`, `zeromodel.video_action_set_final_admin_cli`, `zeromodel.domains.video_action_set.*`, `zeromodel.db.runtime` - confirmed `ModuleNotFoundError` under the current layout (real paths are `zeromodel.persistence.sqlalchemy.video_action_set_final_*_cli` and `zeromodel.video.domains.video_action_set.*`). `scripts/video-final-observe.ps1` and `scripts/video-final-reconstruct.ps1` themselves still invoke the same dead `zeromodel.video_action_set_final_admin_cli` module - this is a production-tooling regression, not merely test debt. It also means `test_video_final_schema_and_scripts.py`'s hostile-input assertion (which exercises these same `.ps1` scripts) likely passes because the module import fails, not because hostile input is actually rejected.
- **Why it matters:** These are opt-in (`integration`/`slow` marked, `workflow_dispatch`-only) so the breakage is invisible in normal CI, but the operator-facing recovery scripts are broken tooling that would fail the moment an operator actually needed them during an incident.
- **Recommended owner:** video-action-set maintainer
- **Recommended implementation stage:** Before next reliance on `scripts/video-final-*.ps1` in an incident; test cleanup can follow
- **Acceptance test:** `.ps1` scripts updated to the current module paths; the three obsolete test files either updated to match or explicitly retired with a reasoned note; `test_video_final_schema_and_scripts.py`'s hostile-input test fails if hostile input were actually accepted (i.e. it's testing real behavior, not an import error).

---
**H3 - Five of six distributions receive no Linux CI validation at all**
- **Affected files:** `.github/workflows/analysis-package.yml`, `observation-package.yml`, `vision-package.yml`, `video-package.yml`, `sqlalchemy-package.yml` (all `runs-on: windows-latest`); `.github/workflows/core-package.yml` (`ubuntu-latest`); `.github/workflows/package-integration.yml` (broken per F2)
- **Exact evidence:** Confirmed by reading every workflow's `runs-on:` line. `package-integration.yml` is the only workflow that would validate all six packages together on Linux, and it always fails at the `validate_release_candidate.py` step (F2).
- **Why it matters:** Every `packages/*/pyproject.toml` declares `Operating System :: OS Independent`, but analysis/observation/vision/video/sqlalchemy currently ship with zero passing Linux CI evidence, while core has zero passing Windows CI evidence.
- **Recommended owner:** CI maintainer
- **Recommended implementation stage:** After F2 is fixed (fixing F2 alone restores Linux coverage for all six via `package-integration.yml`; alternatively add an `ubuntu-latest` leg to each per-package workflow)
- **Acceptance test:** At least one Linux-based workflow run passes end-to-end for each of the five currently Windows-only packages.

---
**H4 - 16 research test files broken by the `examples/arcade_shooter_policy.py` rewrite, plus 3 more with a hidden runtime-only regression**
- **Affected files:** see [post-split-research-health.md](post-split-research-health.md) for the full list; root cause is `examples/arcade_shooter_policy.py`
- **Exact evidence:** The split rewrote this file into a 32-line demo exporting only `ShooterConfig, random_baseline_average, run_policy_episode`; it no longer exports `ACTIONS`, `TinyArcadeShooter`, `compile_policy_artifact`, `state_row_id`, `_action_values`. 16 research test files fail to even collect (4 direct importers, 12 transitive via two intermediate research modules). A further 3 files (`test_arcade_shooter_baseline_comparison.py`, `test_arcade_shooter_example.py`, `test_arcade_shooter_exhaustive.py`) dynamically load the same stub inside each test body, so they collect successfully but will raise `AttributeError` the moment they actually run.
- **Why it matters:** Research failures are explicitly out of scope for "production green," but this is large enough (19 of 30 research test files affected) that it represents a near-total loss of research regression coverage for the arcade action-set work since the split.
- **Recommended owner:** research/video-action-set maintainer
- **Recommended implementation stage:** Next research maintenance pass (not a production blocker)
- **Acceptance test:** `pytest --collect-only research` collects all 30 files with 0 errors, and the 3 hidden-regression files pass when actually executed.

### MEDIUM

- **M1 - `quality-baseline.toml` is 40% stale.** 14 of 35 `legacy_exceptions` entries reference file paths that no longer exist (e.g. `zeromodel/video_action_set_benchmark.py`, `tests/test_video_split_progress_kernel.py` which now lives under `research/video_action_set/tests/`). Inert but unreconciled since the split. *Owner: quality-tooling maintainer. Stage: routine cleanup. Acceptance: every `legacy_exceptions` key resolves to an existing file.*
- **M2 - CI trigger-path gaps.** `python.yml` never triggers on `packages/**` changes at all (its own path list only has the now-nonexistent `zeromodel/**`); `package-integration.yml` never triggers on `tests/**` or `integration_tests/**` changes even though `run_fast_tests.py` collects from both; `core-package.yml` is the only one of the six per-package workflows that omits root `pyproject.toml` from its trigger paths. *Owner: CI maintainer. Stage: routine. Acceptance: editing any file a job actually reads triggers that job.*
- **M3 - Four of seven validation reports cite a stale package-boundary count (118 vs. current 112).** Explainable by vision's later research-module relocation (self-documented in the vision report) but not reconciled across the other four reports. *Owner: documentation owner. Stage: routine. Acceptance: all seven reports state 112 or a dated note explaining the change.*
- **M4 - Per-package CI workflows install ad hoc tool versions instead of `requirements-dev.txt`.** `build twine pytest ruff mypy [numpy]` are pip-installed directly in each of the six per-package workflows rather than reusing `requirements-dev.txt`, creating a version-drift risk between local dev and CI. *Owner: CI maintainer. Stage: routine. Acceptance: workflows install from `requirements-dev.txt` or a pinned lockfile shared with local dev.*
- **M5 - `docs/research/visual-address-phase-one.md` and `visual-address-benchmark.yml` both still recommend/use the broken `pip install -e '.[vision]'`.** Same root cause as F1; lower urgency since this workflow is explicitly a manual research reproduction, not a PR gate. *Owner: research-docs maintainer. Stage: after F1. Acceptance: doc and workflow both use a working install command.*
- **M6 - `integration_tests/test_package_integration_smoke.py` is not auto-tagged `integration` despite being the one true six-distribution smoke test.** `tests/conftest.py`'s `pytest_collection_modifyitems` checks `"integration" in item.path.parts` (exact path-component match); this file's directory is `integration_tests`, a different string, so the check silently misses it. *Owner: test-infrastructure maintainer. Stage: routine. Acceptance: this file is selected by `-m integration` or an equivalent explicit marker.*
- **M7 - Several `tests/` files mix production-unit and cross-package-integration concerns without a marker to separate them** (e.g. `test_video_final_access_kernel.py`, `test_video_final_access_transactions.py` - see [post-split-test-ownership.csv](post-split-test-ownership.csv) "needs splitting" rows), meaning CI cannot selectively include/exclude the cross-package portions the way it can for files that use the `integration` marker correctly. *Owner: video-action-set maintainer. Stage: routine. Acceptance: cross-package test functions carry the `integration` marker or move to `tests/integration/`.*
- **M8 - `pytest-cov` is declared in `requirements-dev.txt` but never invoked anywhere** (`--cov` does not appear in any workflow or script). Not harmful, just an unused declared dependency. *Owner: dev-tooling maintainer. Stage: routine. Acceptance: either wired into a coverage-reporting step or removed.*
- **M9 - Two `tests/integration/` files are misplaced.** `test_video_finalization_failure_injection.py` and `test_video_finalization_historical_evaluator.py` are auto-tagged `integration` purely by directory location but contain no `zeromodel.persistence.sqlalchemy` usage at all - they are single-package (video) production-unit tests. *Owner: video-action-set maintainer. Stage: routine. Acceptance: relocated to `tests/` (or `packages/video/tests/`) without the `integration` marker, or documented reason for the placement kept.*
- **M10 - `tests/conftest.py`'s `INTEGRATION_TEST_FILES` set contains 8 dead filename entries** that don't exist anywhere in the current repository (harmless dead code, but drift). *Owner: test-infrastructure maintainer. Stage: routine cleanup.*
- **M11 - `scripts/check_package_boundaries.py` imports `tomllib` unconditionally with no fallback, so it crashes on Python 3.10** even though every `packages/*/pyproject.toml` declares `requires-python >= 3.10`; does not currently surface because every workflow that calls it pins Python 3.12, but breaks the documented dev-install floor for anyone using 3.10 locally. *Owner: dev-tooling maintainer. Stage: routine. Acceptance: falls back to `tomli` like `validate_release_candidate.py` does (and `tomli` is added to `requirements-dev.txt`).*

### LOW

- **L1 - README's "For development" section omits the dev-tool installs** (`build`, `twine`, `pytest`, `pytest-cov`, `ruff`, `mypy`) - it only shows `pip install -e packages/...` for the six packages, so `python scripts/run_fast_tests.py` immediately below it has no `pytest` unless the reader already has one. *Owner: docs maintainer. Acceptance: README references `requirements-dev.txt` directly.*
- **L2 - `tests/test_visual_result_records.py` is misnamed** relative to its actual content (recovery-manifest checksum verification + an unrelated example run), which could mislead future maintainers searching by name.
- **L3 - `test_video_identity_sql_store.py` tags itself `integration` only via its own `pytestmark`,** inconsistent with sibling SQL-store files that are also listed by filename in `conftest.py`'s `INTEGRATION_TEST_FILES` set - purely a consistency nit, not a functional gap.
- **L4 - Suspected (unconfirmed) duplicate coverage** between numerous `tests/test_video_*_kernel.py` files and `packages/video/tests/` - both exercise the same `zeromodel.video.domains.video_action_set.*` module paths post-split. Flagged for a follow-up cross-check once both suites' contents are compared side-by-side function-by-function (out of scope to fully resolve in this pass; see the full candidate list in [post-split-test-ownership.csv](post-split-test-ownership.csv)).

## Development installation (Area 1) - direct answers

- **Correct editable workspace install command:** `pip install -r requirements-dev.txt` (equivalently `pip install -e ./packages/core -e ./packages/analysis -e ./packages/observation -e ./packages/vision -e ./packages/video -e ./packages/sqlalchemy` plus `build twine pytest pytest-cov ruff mypy`). Verified working end-to-end in a fresh venv.
- **Are all direct dev tools declared?** Mostly. Missing: `tomli` (needed by `check_package_boundaries.py` and as a fallback in `validate_release_candidate.py` on Python <3.11 - see M11). `pytest-cov` is declared but unused (M8).
- **Does Python 3.10 have every required dependency?** No - `check_package_boundaries.py`'s unconditional `import tomllib` fails on 3.10 (M11); not currently exercised by CI since all workflows invoking it pin 3.12.
- **Does documentation still recommend `pip install -e .[dev]`?** No occurrences of `.[dev]` found anywhere in tracked docs. However, `docs/research/visual-address-phase-one.md` still recommends the equally-broken `pip install -e '.[vision]'` (M5), and README's dev section is incomplete rather than wrong (L1).
- **Is the root correctly non-buildable?** Yes - confirmed no `[project]`/`[build-system]` table in root `pyproject.toml`, and `pip install -e .` at root fails with setuptools' flat-layout multi-package error, as intended. The problem is not that the root builds when it shouldn't; it's that several CI workflows and one doc still assume it can (F1, M5).

## Completion report

1. **Baseline SHA:** `c1ce710db50655a6082567fd3f376c3134095ea2` (verified identical to `HEAD`).
2. **Files inspected (representative, not exhaustive):** `requirements-dev.txt`, root `pyproject.toml`, all 6 `packages/*/pyproject.toml`; all 13 files under `.github/workflows/`; `tests/conftest.py`; `scripts/run_fast_tests.py`, `scripts/validate_release_candidate.py`, `scripts/check_quality.py`, `scripts/check_architecture.py`, `scripts/check_package_boundaries.py`, `scripts/code_quality_report.py`, `scripts/create-release.ps1`; `quality-baseline.toml`, `package-boundaries.toml`; `docs/release.md`, `README.md`, `AGENTS.md`, `docs/research/visual-address-phase-one.md`; all 7 `docs/architecture/package-*-validation-1.0.13.md` reports plus `package-public-api-1.0.13.csv` and `package-release-artifacts-1.0.13.json`; all 140 test files across `tests/` (73), `tests/integration/` (9), `integration_tests/` (1), `packages/*/tests/` (27), `research/video_action_set/tests/` (30); every package's `src/zeromodel/.../__init__.py`.
3. **Commands executed:** `git rev-parse`/`git log -S`/`git worktree add` (clean-baseline verification); `pip install -e ".[dev]"` (dry-run failure reproduction); `pip install -r requirements-dev.txt` into an isolated venv; `pytest --collect-only` (root default, root with `--run-integration --run-slow`, `packages/*/tests`, `research` separately); `pytest -q packages/<name>/tests` for all six packages; `ruff format --check` / `ruff check` / `mypy` against the exact governed path lists from `check_quality.py`, plus informative-only runs against `packages/sqlalchemy`; `python scripts/check_quality.py` and `python scripts/check_package_boundaries.py` end-to-end in a clean worktree; `python scripts/check_architecture.py` in a clean worktree to confirm the vacuous pass.
4. **Exact production test collection:** `tests/` + `integration_tests/` (root `testpaths`) = **635 tests total**; **449 selected / 186 deselected** under default markers (no `--run-integration --run-slow`).
5. **Exact package-local test collection:** **144 tests** across `packages/core/tests` (6 files), `packages/analysis/tests` (12 files), `packages/observation/tests` (3 files), `packages/vision/tests` (1 file), `packages/video/tests` (4 files), `packages/sqlalchemy/tests` (1 file) - collected separately since the fast suite does not reach them (F5).
6. **Research collection result:** **25 tests collected, 21 collection errors**, across 30 files under `research/video_action_set/tests/` (the entirety of `research/**/tests/`).
7. **Blocker count:** 8 (F1-F8).
8. **High finding count:** 4 (H1-H4).
9. **Ambiguous ownership decisions:** 0 files landed in a genuine "undecided" bucket across all three classification passes. However, 10 files were flagged "needs splitting" (mixed classification within one file) and roughly 20 `tests/test_video_*` files were flagged as *suspected* (unconfirmed) duplicates of `packages/video/tests/` coverage, pending a direct side-by-side comparison not performed in this pass (L4). Full lists in [post-split-test-ownership.csv](post-split-test-ownership.csv).
10. **Recommended implementation order:**
    1. F1 (root non-buildable install commands) and F2 (Windows-only path in the release validator) - both are one-line-class fixes that unblock the two most-referenced CI workflows.
    2. F3 (vacuous architecture check) and F4 (sqlalchemy quality-gate gap) - both quality-gate integrity fixes, independent of F1/F2.
    3. F5 (fast suite excludes package-local tests) - restores real coverage claims for the "bounded fast suite."
    4. F6 (fake public API manifest) and F7 (wrong golden identity) - documentation/tooling-output correctness, no execution risk.
    5. F8 (no working release orchestration) - needed before any real release is attempted, not before routine CI is green.
    6. H1-H4, then the MEDIUM and LOW items as routine backlog.

## Remediation status (Stage A1)

This is an append-only status update; the findings above are left exactly as originally written. See [post-split-stage-a1-validation.md](post-split-stage-a1-validation.md) for the full account.

Stage A1 ("verification command liveness") addressed **F1, F2, F3, M2, M5, M11, L1**, plus a minimal, scoped exception under **H4** (restoring missing re-exports in `examples/arcade_shooter_policy.py`) that was strictly required to make the `python.yml` `lua-edge` job's Lua-export command executable. All changes were verified by direct execution in this session:

| finding | status | evidence |
|---|---|---|
| F1 (root non-buildable install commands break every job) | **fixed** | `python.yml`, `integration.yml`, `visual-address-benchmark.yml` now install from `requirements-dev.txt` (or explicit `-e packages/...`); `publish-testpypi.yml` and `python.yml`'s `package-build` job now build the six packages explicitly instead of the root |
| F2 (Windows-only path crashes the release validator on Linux CI) | **fixed** | `venv_python()` helper added, unit-tested for both branches; full validator run passed end-to-end on Windows in this session; Linux branch is logic-verified but not executed on a real Linux runner here |
| F3 (architecture check scans zero modules) | **fixed** | `check_architecture.py` now discovers modules from `package-boundaries.toml`'s six source roots, hard-fails on zero modules, and reports "112 production modules inspected" (verified); `check_quality.py` now also runs `check_package_boundaries.py` so the boundary/research-import rules are genuinely enforced by the gate |
| M2 (CI trigger-path gaps) | **fixed for python.yml / package-integration.yml** | `packages/**`, `tests/**`, `integration_tests/**`, `requirements-dev.txt`, and root `pyproject.toml` added where missing; duplicate `scripts/**` entry removed; stale `zeromodel/**` entry removed. Not re-audited for the six per-package workflows (`core-package.yml` etc.), which were out of this stage's scope |
| M5 (visual research workflow/doc use the broken `.[vision]` extra) | **fixed** | both `visual-address-benchmark.yml` and `docs/research/visual-address-phase-one.md` now use `-e packages/core -e packages/observation -e packages/vision`; a second, unrelated import break in `.github/scripts/run_visual_address_smoke.py` was discovered in the process and is documented, not fixed |
| M11 (`check_package_boundaries.py` has no Python 3.10 tomllib fallback) | **fixed** | try/except `tomli` fallback added, matching `validate_release_candidate.py`; `requirements-dev.txt` now declares `tomli>=2; python_version < "3.11"` |
| L1 (README dev-install section omits dev tools) | **fixed** | README now points at `pip install -r requirements-dev.txt` directly |
| H4 (research tests broken by `arcade_shooter_policy.py` rewrite) | **partially addressed (minimal, scoped exception only)** | missing re-exports restored because the `lua-edge` CI job could not otherwise execute; research collection improved from 25/21 to 36/20 as a side effect, but full H4 resolution remains out of scope for this stage |
| F4, F5, F6, F7, F8, H1, H2, H3 (Linux coverage for 5 of 6 packages), and all MEDIUM/LOW items not listed above | **unchanged** | left exactly as originally found; F5 in particular was explicitly deferred to a future "Stage A2" per this stage's own instructions |

Twenty-two new repository-tooling regression tests were added (`tests/test_release_validator_venv_paths.py`, `tests/test_architecture_checker_workspace.py`, `tests/test_workspace_ci_invariants.py`) to guard against these specific regressions recurring. No production or research test was moved, skipped, or altered; no package runtime behavior changed.
