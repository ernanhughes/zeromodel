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

---

## Stage A2.1 addendum: verification hardening

**Starting commit for this addendum:** `fff9c3e694ee3f85f518bbdb1aa3c23b99aedeea` (the Stage A2 work above, already committed to `main` by the time this addendum began - the "uncommitted" note above describes Stage A2's own state at the time it was written, not this addendum's).

This addendum responds to a review of the Stage A2 report identifying three items needing correction/hardening before the stage is considered closed: (1) a claimed "production DTO schema mismatch" that needed verification, (2) fast-suite runtime margin that was too thin to trust, (3) an imprecise "Ruff passes" claim. All three are resolved below.

### 1. The `ObservationDTO` / `OBSERVATION_RECORD_KEYS` finding was corrected, not fixed as a production bug

Full trace of `ObservationDTO.from_record()`/`to_record()` (`packages/video/src/zeromodel/video/domains/video_action_set/observation_dto.py`) proved the original Stage A2 report's claim wrong: `OBSERVATION_RECORD_KEYS` is not supposed to equal every dataclass field. It is the top-level JSON *record* schema, deliberately narrower than the dataclass because five fields are read from inside the record's own `metadata` sub-dict (`benchmark_seed_digest`, `episode_plan_digest`, `provider_observation_descriptor`, `provider_observation_digest`, `operation_chain`) and two more are derived from the optional `pixels` key or supplied as an external keyword argument (`matrix_blob_id`, `final_access_id`). This is correct, intentional production design, not drift.

The actual defect was a stale, incomplete fixture in `tests/test_video_action_set_benchmark.py` (a genuine research file) - its hand-built `fake_records` payload was missing 8 required top-level keys (`benchmark_version`, `generator_version`, `event_type`, `episode_family`, `episode_disposition`, `frame_disposition`, `denominator_class`, `observation_pixel_digest`) that the current record schema requires. That fixture was corrected field-by-field, verifying each requirement against the real, correctly-enforced validation logic at every layer (record keys → ID/split scheme → pixel digest → operation chain → provider descriptor). The fixture now passes every `ObservationDTO`-level validation; it still fails on one further, orthogonal requirement (referential integrity against a real persisted benchmark identity/episode plan, which only exists after `research.benchmarks.video_action_set_benchmark.freeze_benchmark()` runs inside the same test) - a deeper, separate mocking-strategy gap in that one research test, not part of the DTO schema question this addendum was asked to resolve, and not chased further given it does not affect the production gate.

A precise, permanent regression test was added at `packages/video/tests/test_observation_record_schema.py` (production-owning package, not research): it asserts `OBSERVATION_RECORD_KEYS` is a strict subset of the dataclass fields, and that every excluded field is explicitly accounted for in one of two named, documented buckets (`METADATA_EMBEDDED_FIELDS`, `DERIVED_OR_EXTERNAL_FIELDS`). If a future field is added to `ObservationDTO` without a decision about which bucket it belongs to, this test fails loudly instead of silently passing or silently drifting.

**Verdict: not a production bug.** Corrected in the record above; no production code semantics changed (the one line touched in `packages/sqlalchemy/src/.../db/stores/video_action_set.py` in Stage A2 - the `CursorResult` narrowing - remains the only production behavior change across A2 and this addendum).

### 2. Fast-suite runtime: profiled, and the real cost driver fixed rather than the timeout raised

`pytest --durations=40` with the exact canonical fast-suite invocation (all 8 test roots, `--run-integration`, `-m "not slow and not external and not research"`) identified the actual cost driver precisely:

- `tests/integration/test_video_finalization_package_boundary.py::test_video_wheel_alone_carries_no_sqlalchemy_finalization_capability` - **15.75s alone** (it builds a real wheel and creates a venv from scratch - never a "fast, bounded" operation by definition).
- `tests/integration/test_video_finalization_cli_scripts.py::test_powershell_admin_wrapper_propagates_json_and_exit_code` (2 parametrized cases) - ~3.2s each, spawning a separate PowerShell process per case - an environment-specific dependency (a `pwsh`/PowerShell executable on `PATH`), not a computation cost.

Neither was optimized in place; both were reclassified per the taxonomy's own rules (own criteria: "tests that are inherently expensive should receive the slow marker... rather than remaining fast solely because they once completed under two minutes"):
- the wheel-building test gained `@pytest.mark.slow` (in addition to its existing `integration` marker from directory placement);
- the PowerShell-invoking test gained `@pytest.mark.external` (environment-specific infrastructure, matching that marker's own definition exactly).

Result, profiled with `--durations=15` after reclassification: **no single test exceeds 2.3s**, and three consecutive full `python scripts/run_fast_tests.py` runs measured:

| run | duration | result |
|---|---|---|
| 1 | 75.34s | 795 passed, 0 failed |
| 2 | 73.80s | 795 passed, 0 failed |
| 3 | 73.84s | 795 passed, 0 failed |

Slowest run 75.34s, median ~74s - both comfortably inside the requested acceptance target (slowest < 100s, median < 90s), and a real ~45s improvement over the pre-fix 122.84s worst case observed during profiling. A regression test (`tests/test_fast_suite_runtime_stability.py`) asserts both markers stay in place so this cannot silently regress back.

### 3. Ruff claim corrected: governed-gate result vs. whole-repository result, stated separately and precisely

Both commands were run exactly as specified, unscoped:

```
ruff check .          -> FAILS: 8 pre-existing E402 findings
                          (6 in examples/arcade_visual_video_discriminative_evidence_benchmark.py,
                           2 in examples/render_signs_demo.py)
ruff format --check . -> FAILS: 141 files would be reformatted
```

Both files with lint findings were confirmed untouched by any Stage A1/A2/A2.1 change (`git diff --stat HEAD` on each returns empty) - pre-existing, not a regression introduced here. One genuine regression *was* found and fixed during this check: `packages/video/tests/test_video_policy_reader_contracts.py` (new in Stage A2) needed reformatting; it has been reformatted and re-verified.

The governed gate (`scripts/check_quality.py`'s exact `FORMAT_LINT_PATHS`) is separately confirmed 100% clean for both commands. The prior report's unqualified "Ruff passes" line is superseded by this precise statement. A regression test (`tests/test_ruff_scope_claims.py`) now (a) asserts the governed paths pass both commands, and (b) asserts the whole-repo lint run's offender set never grows beyond the two currently-known, currently-out-of-scope files - so a *new* file added anywhere in the repo with an E402-style violation will fail this test, while the two pre-existing ones remain explicitly, visibly tolerated rather than silently ignored.

### A2.1 validation commands run

```
python -m mypy packages/sqlalchemy/src packages/sqlalchemy/tests -> Success: no issues found in 13 source files
python scripts/check_package_boundaries.py                       -> Package boundary check passed: 112 production modules
python scripts/check_architecture.py                             -> Architecture check: passed (112 production modules inspected)
python scripts/check_quality.py                                  -> Quality checks passed
python scripts/run_fast_tests.py (x3)                             -> 795/795/795 passed, 0 failed, 75.34s/73.80s/73.84s
python -m pytest packages/video/tests -q                          -> 24 passed
python -m pytest packages/sqlalchemy/tests -q                     -> 6 passed
python -m pytest integration_tests -q                             -> 1 passed
python scripts/validate_release_candidate.py                     -> Release candidate validation passed (211 public symbols; test-layer report: 538 fast, 107 integration, all 6 packages' wheel smoke true)
git diff --check                                                 -> clean (line-ending warnings only, exit 0)
ruff check . / ruff format --check .                             -> fail unscoped as documented above; governed gate clean
```

### A2.1 files changed

`packages/sqlalchemy/tests/test_sqlalchemy_package_isolation.py` was not touched in this addendum (already fixed in A2). New/changed in this addendum: `tests/test_video_action_set_benchmark.py` (fixture repaired), `packages/video/tests/test_observation_record_schema.py` (new), `tests/integration/test_video_finalization_package_boundary.py` (`@pytest.mark.slow` added), `tests/integration/test_video_finalization_cli_scripts.py` (`@pytest.mark.external` added, reformatted), `tests/test_fast_suite_runtime_stability.py` (new), `tests/test_ruff_scope_claims.py` (new), `packages/video/tests/test_video_policy_reader_contracts.py` (reformatting fix only, no logic change).

### A2.1 confirmation

Nothing was published, tagged, or version-bumped. No commit or push was performed by this addendum - all changes remain in the working tree pending your explicit go-ahead, on top of the already-committed `fff9c3e`.
