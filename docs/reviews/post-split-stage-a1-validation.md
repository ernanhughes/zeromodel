# Post-split remediation — Stage A1: verification command liveness

**Baseline commit:** `26d7042e02b6e810a2d47a141c6fbe0f3e9d2dd2`
**Final commit:** not yet committed at the time of writing — all changes described below are staged in the working tree, pending an explicit commit decision.
**Objective:** make every active development, CI, and verification command executable against the six-distribution workspace (findings F1, F2, F3, M2, M5, M11, L1 from [post-split-main-audit.md](post-split-main-audit.md)). F4–F8 and H2–H4 were left untouched except where a minimal change was strictly required to make an affected command executable (see "Minimal H4 exception" below). No test was moved, no package behavior changed, no test was skipped to obtain green output.

## 1. Canonical development installation

`requirements-dev.txt` already installed all six packages editable (`-e ./packages/core` ... `-e ./packages/sqlalchemy`) plus `build twine pytest pytest-cov ruff mypy`. Added:

```
tomli>=2; python_version < "3.11"
```

so the Python-3.10 floor declared by every `packages/*/pyproject.toml` is actually satisfiable by the dev tooling (see §6). `README.md`'s "For development" section now reads:

```bash
python -m pip install -r requirements-dev.txt
python scripts/run_fast_tests.py
python scripts/validate_release_candidate.py
```

replacing the previous `pip install -e packages/core -e packages/analysis ...` enumeration (functionally equivalent for the packages, but omitted the dev tools, which is why it's now `requirements-dev.txt` directly).

## 2-3. Replaced stale root installation commands / repaired `python.yml`

Every occurrence of `pip install .[dev]`, `-e .[dev]`, `-e .[release]`, `-e '.[vision]'` in active workflows was replaced:

| workflow | before | after |
|---|---|---|
| `python.yml` (`quality`) | `pip install -q .[dev]` | `pip install -q -r requirements-dev.txt` |
| `python.yml` (`package-tests`, matrix 3.10-3.12) | `pip install -q -e .[dev]` | `pip install -q -r requirements-dev.txt` |
| `python.yml` (`lua-edge`) | `pip install -q -e .[dev]`, then `pytest tests/test_lua_policy.py` (file does not exist) | `pip install -q -r requirements-dev.txt`, then `pytest packages/core/tests/test_policy_lookup_lua.py` (the real post-split home of this fixture) |
| `python.yml` (`package-build`) | `pip install -q -e .[release]`; `python -m build` (root, unbuildable) | `pip install -q build twine`; explicit `python -m build packages/<name>` + `python -m twine check` loop over all six packages |
| `integration.yml` | `pip install -e .[dev]` | `pip install -r requirements-dev.txt` (pytest invocation left byte-for-byte unchanged; verified `tests/test_test_tier_commands_kernel.py` still passes, which asserts this exact command string) |
| `visual-address-benchmark.yml` | `pip install -e '.[vision]'` | `pip install -e packages/core -e packages/observation -e packages/vision` (plus the existing pinned CPU torch/torchvision install, which stays workflow-local and is not added to any package's production dependencies) |

`python.yml`'s trigger paths were deduplicated and repointed: `zeromodel/**` (a directory that does not exist in this repo) removed, replaced with `packages/**`; added `integration_tests/**` and `requirements-dev.txt`; removed the duplicate `scripts/**` entry. No publishing was added anywhere.

**Minimal H4 exception (examples/arcade_shooter_policy.py):** the `lua-edge` job's "Export arcade policy for Lua" step runs `examples/lua_edge_policy.py`, which imports `ACTIONS, compile_policy_artifact` from `examples/arcade_shooter_policy.py` — the same symbols the audit's H4 finding identified as missing after that file was rewritten into a 32-line demo during the split. Per your instruction to prefer restoring the missing symbols for H4, and because this specific command could not be made executable any other way, `examples/arcade_shooter_policy.py` now re-exports `ACTIONS`, `TinyArcadeShooter`, `compile_policy_artifact`, `state_row_id`, `_action_values` directly from `zeromodel.video.arcade_policy.model`, where all five were already implemented (this was a missing re-export, not missing logic — no behavior was invented). This is a minimal, targeted fix, not a resolution of H4 at large: research collection improved from 25 collected/21 errors to **36 collected/20 errors** as a side effect, but the remaining 20 research failures (including one newly-visible, unrelated `ImportError: cannot import name 'COOLDOWN_BLOCKED_VALUE' from 'examples.arcade_visual_sign_reader'` surfaced while investigating `test_video_local_correlation.py`) were left untouched, as they are outside this stage's scope.

## 4. Cross-platform release validator

`scripts/validate_release_candidate.py`:
- Added `venv_python(venv, *, is_windows=None)`, defaulting to `os.name == "nt"`, returning `Scripts/python.exe` or `bin/python`. `install_and_probe()` now calls this instead of hardcoding `Scripts/python.exe`.
- Replaced the hardcoded `venv / "Lib" / "site-packages"` prefix check with `is_beneath(path, root)`, a platform-neutral `Path.resolve().relative_to()` containment check, used to verify every probed module import resolves inside the clean venv (not the checkout) regardless of OS-specific site-packages layout.
- Added `tests/test_release_validator_venv_paths.py` (6 tests): Windows interpreter path, POSIX interpreter path, host-default path, checkout-path rejection, venv-import acceptance (both `Lib/site-packages` and `lib/pythonX.Y/site-packages` layouts). All pass.
- **Ran the full validator end-to-end on this machine (Windows):** builds all six packages, `twine check` on each, creates a clean venv, installs all six wheels, `pip check`, import-location probe, root-import-rejection assertion, manifest write — **"Release candidate validation passed."** This is real Windows execution evidence, not a dry run.
- The Linux (`os.name != "nt"`) branch was verified via the unit tests above (returns `bin/python`, the standard CPython venv layout on POSIX) but was **not** executed against an actual Linux runner in this session — no Linux environment was available here. `package-integration.yml`'s `ubuntu-latest` matrix is the intended place this gets proven in CI.

## 5. Architecture validation

Chose **Option A** (workspace-aware `check_architecture.py`), reusing `package-boundaries.toml` as the shared source-root authority (the same manifest `check_package_boundaries.py` already reads correctly):
- `PACKAGE_ROOT = REPO_ROOT / "zeromodel"` (a directory that doesn't exist post-split) replaced with `workspace_source_roots()`, which reads all six `packages/*/src` roots from `package-boundaries.toml`.
- `module_name_for_path` now takes an explicit `source_root` parameter instead of always resolving relative to `REPO_ROOT`, so discovered module names match real import paths (e.g. `zeromodel.persistence.sqlalchemy.db.orm.base`, not a path relative to a non-existent root).
- `main()` now hard-fails (`raise SystemExit`) if zero modules are discovered, and reports the inspected count on success.
- `ImportEdge`, `collect_import_edges()`, and `forbidden_edge_violations()` were **not** changed — `research/video_action_set/tests/test_video_benchmark_facade.py` and `test_video_verification_closure_kernel.py` construct `ImportEdge` and call `forbidden_edge_violations()` directly with exactly 3 fields; changing that surface would have broken those (already-broken-for-unrelated-reasons, out-of-scope) tests further. Verified with a regression test (`test_forbidden_edge_violations_still_accepts_the_three_field_import_edge`).
- **Known limitation, documented rather than silently left:** the video-action-set-specific layering rules in `scripts/architecture_rules/*.py` (`video_action_set.py`, `video_action_set_modules.py`, `video_instrument_shell.py`, `video_science_layers.py`, `video_verification_layers.py`) hardcode pre-split module-name prefixes (e.g. `zeromodel.domains.video_action_set`, `zeromodel.db.orm`, `zeromodel.video_action_set_benchmark`) that no longer match any real module under the current namespace (`zeromodel.video.domains.video_action_set.*`, `zeromodel.persistence.sqlalchemy.db.orm.*`). These rules still run (nothing was deleted) but cannot currently produce a true positive against production code, since the concepts they reference were either renamed or relocated to `research/` during the split — the latter category is now redundantly, and correctly, caught by `check_package_boundaries.py`'s blanket "no production import of `research.*`" rule instead. Fully migrating these ~40 hardcoded constants was judged out of scope for a "minimal, verifiable" stage, since it requires domain knowledge of each rule's original intent that isn't safe to fabricate; flagged here as a candidate for its own follow-up rather than silently left unmentioned.
- Instead of duplicating `check_package_boundaries.py`'s already-correct cross-package boundary and research-import rules inside `check_architecture.py`, **`scripts/check_quality.py` now also runs `scripts/check_package_boundaries.py`** as its own "Package boundaries" step, alongside the repaired "Architecture" step. This is how the repository-wide gate (used by `python.yml` and `package-integration.yml`) ends up genuinely enforcing "reject research imports" and "reject forbidden sibling-package dependencies," without reinventing logic that already worked.
- Added `tests/test_architecture_checker_workspace.py` (6 tests): discovers >100 modules, covers all six source-root prefixes, hard-fails with zero configured modules (via a monkeypatched empty `package-boundaries.toml`), detects a deliberately constructed local import cycle, stays clean on an acyclic graph, and the `ImportEdge` regression guard above.
- **Verified:** `python scripts/check_architecture.py` now reports `Architecture check: passed (112 production modules inspected)`, confirmed against a clean git worktree at the baseline commit (matching `check_package_boundaries.py`'s own count exactly).

## 6. Python 3.10 compatibility

`scripts/check_package_boundaries.py` had an unconditional `import tomllib` (Python 3.11+ stdlib only) with no fallback, unlike `scripts/validate_release_candidate.py` (which already had one). Added the same pattern:

```python
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
```

`requirements-dev.txt` now supplies `tomli` for Python <3.11 (see §1). No package runtime requirements were changed.

## 7. Package-integration workflow

`package-integration.yml`:
- Install step switched from ad hoc `pip install build twine pytest ruff mypy` to `pip install -r requirements-dev.txt`, so the Python 3.10 leg actually has `tomli` available for both `check_package_boundaries.py` and `validate_release_candidate.py`.
- Trigger paths extended to include `tests/**` (the root fast suite's other collection root), `requirements-dev.txt`, and `pyproject.toml` (root tooling/pytest/ruff/mypy configuration).
- The final step (`validate_release_candidate.py`) is unchanged in structure — still builds real wheels and installs them into a separate, disposable venv (`build/full-integration-venv`); nothing was converted to editable installs.
- **Cannot be executed on real GitHub-hosted Ubuntu runners from this session** (no Linux CI access here); confidence that all three Python legs now reach the final validation step rests on: (a) the Windows end-to-end run in §4 exercising the identical code path with `is_windows=False` logic unit-tested separately, and (b) `tomli` now being installed via `requirements-dev.txt` for the 3.10 leg specifically.

## 8. Research workflow installation

`visual-address-benchmark.yml` and `docs/research/visual-address-phase-one.md` both replaced `pip install -e '.[vision]'` with `pip install -e packages/core -e packages/observation -e packages/vision` (vision's actual dependency closure). Torch/torchvision remain workflow/doc-local research dependencies, not added to any package's `pyproject.toml`.

**Discovered, not fixed (documented rather than silently skipped):** `.github/scripts/run_visual_address_smoke.py`, which this workflow runs after installing dependencies, imports `zeromodel.visual_encoder`, `zeromodel.visual_experiment`, `zeromodel.visual_precomputed`, `zeromodel.visual_retrieval` — flat module paths that do not exist anywhere in the current repository under that name (the real modules live at `research/visual/visual_*.py`). This is a second, independent break in the same workflow, outside the "repair the install command" scope of this stage, and was not touched. The workflow is `workflow_dispatch`-only and explicitly documented in its own header as "not routine product or pull-request gates," so this does not block any PR or push today, but it means the workflow still cannot complete a real run even after this stage's fix. Recommended for its own follow-up.

## 9. TestPyPI workflow safety

`publish-testpypi.yml` was rewritten (option: convert to non-publishing six-package build validation): the actual `pypa/gh-action-pypi-publish` step and `id-token: write` permission were removed; the workflow now only builds and `twine check`s all six `packages/*` distributions on `workflow_dispatch`, with a header comment explaining that six-package TestPyPI/PyPI publication is deferred to a dedicated release-process stage (audit finding F8). No publish action of any kind remains in this workflow.

## 10. Tests added

| test file | covers |
|---|---|
| `tests/test_release_validator_venv_paths.py` | `venv_python()` (Windows/POSIX/host-default), `is_beneath()` (checkout rejection, venv acceptance on both site-packages layouts) |
| `tests/test_architecture_checker_workspace.py` | module discovery >0 and across all six roots, zero-module hard-fail, cycle detection (positive and negative), `ImportEdge`/`forbidden_edge_violations` backward-compatibility guard |
| `tests/test_workspace_ci_invariants.py` | `requirements-dev.txt` six-package + tomli-marker expectations, no active workflow invokes root extras or root build, root `pyproject.toml` still has no `[project]`/`[build-system]` table, `package-integration.yml`/`python.yml` install and trigger-path invariants, tomllib/tomli fallback present in both scripts, `publish-testpypi.yml` makes no publish claim |

No production or research behavioral test was moved or modified.

## 11. Validation run (this session, on Windows; see per-section notes above for Linux-specific caveats)

```
python -m pip install -r requirements-dev.txt        -> succeeded, all six packages + tomli present
python scripts/check_package_boundaries.py            -> "Package boundary check passed: 112 production modules"
python scripts/check_architecture.py                  -> "Architecture check: passed (112 production modules inspected)"
python scripts/check_quality.py                        -> "Quality checks passed" (Formatting/Linting/Typing/Architecture/Package boundaries/Quality limits all passed)
python scripts/run_fast_tests.py                        -> "471 passed, 186 deselected" in 39.94s (budget 120s) — up from 449 passed pre-stage; +22 = the 22 new tests added in this stage
python scripts/validate_release_candidate.py            -> "Release candidate validation passed" (full six-package build/check/clean-install/probe/manifest)
pytest packages/core|analysis|observation|vision|video|sqlalchemy/tests -> 143 passed, 1 skipped (Lua interpreter not installed locally — pre-existing, expected skip)
```

Ruff format/lint were also run directly against every file touched or added in this stage (even where not part of the governed gate, e.g. root `tests/`) and are clean.

Note: running the fast suite (which includes `tests/test_analyze_package_inventory.py`) regenerated several `docs/architecture/*` inventory artifacts (`package-inventory-1.0.13.md`, `package-dependency-findings-1.0.13.md`, `package-import-graph-1.0.13.json`, `package-module-map-1.0.13.csv`) as a side effect of that pre-existing test's own design (it writes real inventory output, not to a temp path). The regenerated content accurately reflects this stage's actual changes (e.g. the "tooling" module count moved from 98 to 101, matching the 3 new test files). `docs/architecture/package-release-artifacts-1.0.13.json` was likewise regenerated by `validate_release_candidate.py` itself, as designed.

## 12. Remaining audit findings (unchanged by this stage)

F4 (sqlalchemy quality-gate coverage), F5 (fast suite excludes package-local tests — explicitly deferred to "Stage A2" per this stage's own instructions), F6 (public-API manifest placeholder), F7 (wrong golden identity value), F8 (release orchestration rewrite), H1 (claims-audit governance gate), H2 (obsolete finalization tests + `.ps1` scripts), H3 (five of six packages lack Linux CI — partially improved by this stage's `package-integration.yml` fixes, but not independently re-verified on a real Linux runner), and all MEDIUM/LOW findings remain exactly as described in `post-split-main-audit.md`. Two new, narrowly-scoped issues were discovered and documented (not fixed) during this stage: the unrelated `COOLDOWN_BLOCKED_VALUE` import failure in `examples/arcade_visual_sign_reader.py` (§2-3 above) and the flat `zeromodel.visual_*` import failures in `.github/scripts/run_visual_address_smoke.py` (§8 above).

## Completion report

1. **Final commit SHA:** not committed yet — pending your go-ahead.
2. **Workflows changed:** `python.yml`, `package-integration.yml`, `integration.yml`, `publish-testpypi.yml`, `visual-address-benchmark.yml`.
3. **Removed root-install commands:** `pip install .[dev]` / `-e .[dev]` (python.yml x3, integration.yml), `-e .[release]` (python.yml package-build), `-e '.[vision]'` (visual-address-benchmark.yml). `python -m build` against the repository root removed from `python.yml`'s package-build job (replaced with an explicit six-package loop).
4. **Canonical development command:** `python -m pip install -r requirements-dev.txt`.
5. **Architecture module count:** 112 (matches `check_package_boundaries.py` exactly; previously 0).
6. **Cross-platform release-validator results:** Windows — executed end-to-end, passed. Linux — path-selection logic unit-tested (6/6 passing), not executed on a real Linux runner in this session.
7. **Python 3.10-3.12 results:** not independently executed per-version in this session (single local Python 3.11 environment available); `tomli` availability for 3.10 confirmed via `requirements-dev.txt` marker and regression test.
8. **TestPyPI workflow disposition:** converted to a non-publishing six-package build-check workflow; publishing explicitly deferred to a future release-process stage; `id-token: write` removed.
9. **Focused tooling-test result:** 22 new tests added across 3 files, all passing (`31 passed` when run together with the pre-existing release-metadata/package-boundaries/tier-commands tests they neighbor).
10. **Remaining blockers for Stage A2:** F4, F5 (explicitly deferred here), F6, F7, F8, H1-H4, plus the two newly-discovered-but-undocumented-until-now research import breaks noted in §12 above.
