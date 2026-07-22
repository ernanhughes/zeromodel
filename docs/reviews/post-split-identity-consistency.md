# Post-split identity consistency report

Baseline: `c1ce710db50655a6082567fd3f376c3134095ea2`. Scope: the 7 validation reports under `docs/architecture/` (`package-core-validation-1.0.13.md`, `package-analysis-validation-1.0.13.md`, `package-observation-validation-1.0.13.md`, `package-vision-validation-1.0.13.md`, `package-video-validation-1.0.13.md`, `package-sqlalchemy-validation-1.0.13.md`, `package-integration-validation-1.0.13.md`) plus `docs/architecture/package-public-api-1.0.13.csv`.

## Conflicting golden identity value (BLOCKER)

`package-core-validation-1.0.13.md` and `package-integration-validation-1.0.13.md` both describe the same fixture - "the core artifact kernel" golden artifact ID - and give **two different SHA-256-shaped values**:

- `package-core-validation-1.0.13.md`: `32f8013789e4ff463569e2ccbbdc8c3802bc42c6edeb8ceb361afca9a6025db1`
- `package-integration-validation-1.0.13.md`: `32f801671139b73e349c756570c27c06d39c422a4d9a277782e1c997a473083b`

Ground truth: `packages/core/tests/test_artifact_kernel.py:16-18` pins `GOLDEN_SAMPLE_ARTIFACT_ID = "32f801671139b73e349c756570c27c06d39c422a4d9a277782e1c997a473083b"` - matching the **integration** report, not the core report.

`git log --all -S "32f8013789e4ff463569e2ccbbdc8c3802bc42c6edeb8ceb361afca9a6025db1"` across the full repository history returns exactly one hit: the markdown line itself. The pre-split root test already carried the *other* (correct) value several commits earlier. This means the core-validation report's claim is not a value that drifted after being true - **it was never true**. No test anywhere in the repository asserts it.

**Severity:** BLOCKER. A "golden identity" claim in a validation report that is documentation-only, contradicted by a sibling report, and never matched by any test is exactly the failure mode identity-consistency review exists to catch.

### Resolution (Stage A2)

The enforced test is authoritative: `packages/core/tests/test_artifact_kernel.py:16-18` (`GOLDEN_SAMPLE_ARTIFACT_ID`), asserted at line 58, has always pinned `32f801671139b73e349c756570c27c06d39c422a4d9a277782e1c997a473083b` for the "sample artifact" fixture (a fixed `LayoutRecipe` + `ScoreTable` built once at module scope). This is the *same fixture* the core-validation report describes - not a different, imprecisely-named one - so the resolution is case (a): the same fixture, one incorrect documented value, corrected to match the test rather than the other way around. The digest was not regenerated or altered in any way; only the documentation was corrected.

`docs/architecture/package-core-validation-1.0.13.md`'s "Golden Identity Results" section now records, for this fixture: fixture name, fixture version (current, no historical predecessor), producing test, expected digest, owning test location, and status (current/enforced) - and explicitly notes the old, incorrect value as a corrected historical error rather than silently dropping it. `docs/architecture/package-integration-validation-1.0.13.md` already had the correct value and required no change. A new regression test, `tests/test_identity_documentation_consistency.py`, now enforces that both documents agree with `GOLDEN_SAMPLE_ARTIFACT_ID` going forward.

## Documentation-only quality claim (sqlalchemy)

`package-sqlalchemy-validation-1.0.13.md` states the "Repository quality gate: `python scripts/check_quality.py` ... passed" and describes a "focused" ruff/mypy pass over "12 source files," presented in the same register as the equivalent, gate-backed claims in the analysis and observation reports.

This is misleading: `scripts/check_quality.py`'s `FORMAT_LINT_PATHS`/`TYPING_PATHS` (lines 8-34) and root `pyproject.toml`'s `[tool.mypy] mypy_path` (line 25) omit `packages/sqlalchemy` entirely (cross-referenced against [post-split-quality-coverage.csv](post-split-quality-coverage.csv)). `check_quality.py` passing today is true, but it is vacuously true for sqlalchemy - the script never inspects that package's code, so it cannot fail on it regardless of what the code looks like. Direct execution confirms `mypy packages/sqlalchemy/src` independently produces 17 errors (including one genuine bug: `db/stores/video_action_set.py:1070` - `Result[Any]` has no attribute `rowcount`) that the "focused" ad hoc commands in the report's own text would have had to be run manually and separately from the governed gate to get a clean result, since the governed gate never touches this code at all.

**Severity:** HIGH. The report's phrasing claims equivalent assurance to packages whose quality actually is gate-enforced; it is not, and the gap is currently masking a real mypy failure.

### Resolution (Stage A2)

`packages/sqlalchemy/src` and `packages/sqlalchemy/tests` are now in `scripts/check_quality.py`'s `FORMAT_LINT_PATHS`/`TYPING_PATHS`/`QUALITY_LIMIT_PATHS` and root `pyproject.toml`'s `mypy_path`. The real mypy bug this gap was masking (`db/stores/video_action_set.py:1070`) is fixed with a runtime-validated `isinstance(result, CursorResult)` narrowing rather than a blanket ignore; `python -m mypy packages/sqlalchemy/src packages/sqlalchemy/tests` now passes with zero errors, verified directly. The report's claim is no longer misleading - it is now backed by the same governed gate as every other package. See `docs/reviews/post-split-stage-a2-validation.md` for full evidence.

## Stale-but-explainable module-count drift (not a defect)

`package-core-validation-1.0.13.md`, `package-analysis-validation-1.0.13.md`, `package-observation-validation-1.0.13.md`, and the pre-closure section of `package-vision-validation-1.0.13.md` all state "Package boundary check passed: 118 production modules." `package-video-validation-1.0.13.md` and `package-sqlalchemy-validation-1.0.13.md` state 112. Running `scripts/check_package_boundaries.py` against the current repo returns **112**.

This is explained by the vision report's own "Vision Closure" section, which documents 6 research modules being moved out of the production tree after the earlier four reports were written. It is not a contradiction of fact, but four now-stale point-in-time numbers that no longer match the current repo; an auditor who doesn't read the vision report's closure section could mistake this for an unexplained discrepancy.

**Severity:** LOW/MEDIUM - correct at time of writing, stale now, self-documented by one of the seven reports but not reconciled across the other four.

## Values independently re-verified with no conflict found

- Every package's `__all__` list, as printed in its validation report, matches the current `packages/<name>/src/zeromodel/.../__init__.py` exactly (core 52 symbols, analysis 87, observation 12, vision 18, video 32, sqlalchemy 10 - 211 total).
- Every package's declared dependency list matches its current `pyproject.toml`.
- Source test counts (`pytest -q packages/<name>/tests`) reproduce exactly against every report's claimed count: core 20 passed/1 skipped, analysis 65 passed, observation 32 passed, vision 10 passed, video 10 passed, sqlalchemy 6 passed.
- `package-integration-validation-1.0.13.md`'s own identity value (`3ce8dd265b949b3b26ebcd602c8b572c248b25c5bafdd13b459a1ab739533e4a` for the cross-package smoke artifact) is present and asserted at `integration_tests/test_package_integration_smoke.py:100-101` and is not contradicted anywhere else.
- All six wheel/sdist SHA-256 digests in the integration report's artifact table were recomputed against the existing build artifacts in `packages/*/dist/` and match `docs/architecture/package-release-artifacts-1.0.13.json` byte-for-byte.
- Wheel content claims independently re-verified by rebuilding/inspecting archives: vision's post-closure wheel (7 total entries: `__init__.py`, `visual.py`, `visual_policy.py` + 4 dist-info files) and sqlalchemy's wheel (16 files) both match their reports exactly.

## Public API manifest cross-check (`docs/architecture/package-public-api-1.0.13.csv`)

**Verdict: the CSV is not a real public-API manifest.** `scripts/validate_release_candidate.py`'s `write_public_exports()` (lines 311-337) writes exactly one row per package with the literal string `"__all__"` hardcoded into the `exported_symbol` column - never the actual exported names. Confirmed by reading the function body and the live CSV (6 data rows total, one per package).

| package | CSV rows | real `__all__` count | gap |
|---|---|---|---|
| zeromodel (core) | 1 (placeholder) | 52 | 51 |
| zeromodel-analysis | 1 (placeholder) | 87 | 86 |
| zeromodel-observation | 1 (placeholder) | 12 | 11 |
| zeromodel-vision | 1 (placeholder) | 18 | 17 |
| zeromodel-video | 1 (placeholder) | 32 | 31 |
| zeromodel-sqlalchemy | 1 (placeholder) | 10 | 9 |
| **Total** | **6** | **211** | **205** |

The CSV's `namespace` and `owning_module` columns are correct (they are hardcoded per-package constants that happen to match reality), but the `exported_symbol` column - the entire reason the file exists - captures roughly 2.8% of the real public surface (6 marker rows instead of 211 per-symbol rows). Every individual `.md` validation report's own `__all__` listing is accurate; only the machine-generated CSV, which is supposed to be the durable, tool-verified source of truth, is a placeholder.

**Severity:** BLOCKER. A manifest that is consulted by tooling or reviewers to answer "is symbol X part of the public API of package Y" currently cannot answer that question at all.

## Ranked reproducibility summary

| report | reproducible? | headline issue |
|---|---|---|
| package-core-validation-1.0.13.md | NO | Golden artifact-kernel ID is wrong and always was; also cites stale 118-module count |
| package-sqlalchemy-validation-1.0.13.md | PARTIALLY | Quality-gate claim implies governed coverage that does not exist |
| package-analysis-validation-1.0.13.md | MOSTLY | Only the stale 118-module count fails to reproduce |
| package-observation-validation-1.0.13.md | MOSTLY | Only the stale 118-module count fails to reproduce |
| package-vision-validation-1.0.13.md | YES | Pre-closure count is explicitly superseded within the same document |
| package-video-validation-1.0.13.md | YES | No discrepancies found |
| package-integration-validation-1.0.13.md | YES | No discrepancies found; holds the correct golden value |
| package-public-api-1.0.13.csv | NO | Placeholder rows, not a real per-symbol manifest (211 real symbols vs. 6 rows) |
