# Post-split research health report

Baseline: `c1ce710db50655a6082567fd3f376c3134095ea2`. Scope: `research/video_action_set/tests/` (30 files - this is the entire `research/**/tests/` tree; `find research -path "*/tests/*" -name "test_*.py"` returns nothing outside `research/video_action_set`).

Collection was run separately from the production suites, with all six split packages installed editable (`pip install -r requirements-dev.txt` into an isolated venv), via:

```
pytest -q --collect-only research
```

Result: **25 tests collected across 9 files, 21 collection errors across 21 files.** Research failures are not treated as production failures anywhere in this report; none of the 21 broken files were repaired.

## Root causes, in order of blast radius

### 1. Stale import: `examples/arcade_shooter_policy.py` no longer exports `ACTIONS` / `compile_policy_artifact` (16 files)

During the split, `examples/arcade_shooter_policy.py` was rewritten into a 32-line demo script that imports only `ShooterConfig, random_baseline_average, run_policy_episode` from `zeromodel.video.arcade_policy.model`. It no longer defines or re-exports `ACTIONS`, `TinyArcadeShooter`, `compile_policy_artifact`, `state_row_id`, or `_action_values`.

- **4 files import the stub directly** and fail immediately: `test_video_action_equivalence_top1.py`, `test_video_action_equivalence_evidence_closure.py`, `test_video_local_correlation.py`, `test_video_policy.py`.
- **12 files fail transitively** through two intermediate research modules that themselves import the missing symbols:
  - `examples/arcade_visual_video_discriminative_evidence_benchmark.py:23` -> feeds `test_video_discriminative_benchmark.py`, `test_video_discriminative_measurement_audit.py`, `test_video_discriminative_representation_audit.py`, `test_video_discriminative_v2_benchmark.py`, `test_video_discriminative_v2_integrity.py`, `test_video_discriminative_v2_selection.py`, `test_video_discriminative_v3_benchmark.py`, `test_video_discriminative_v3_self_retrieval.py` (8 files)
  - `research/video_action_set/arcade_visual_action_equivalence_audit.py:13` -> feeds `test_video_action_equivalence_inventory.py`, `test_video_action_equivalence_bounded_measurements.py`, `test_video_action_equivalence_audit.py`, `test_video_policy_reachability.py` (4 files)

Classification: **stale import**, all 16. This is a reorg artifact of the split, not a behavioral research regression - the example rewrite simply never accounted for these research consumers.

### 2. Syntax error: `test_video_benchmark_facade.py` (1 file, plus 2 latent bugs underneath)

Line 14 reads `xfrom research.video_action_set.video_action_set_cli import main` - a literal typo (`xfrom`). Classification: **syntax error**.

Two additional bugs are stacked underneath and would surface immediately once the typo is fixed:
- Line 21 references `artifact_io._write_json` but `artifact_io` is never imported anywhere in the file -> `NameError`.
- `REPO_ROOT = Path(__file__).resolve().parents[1]` resolves to `research/video_action_set` rather than the true repo root, breaking both `test_benchmark_facade_contains_no_function_implementations` (looks for a nonexistent nested path) and the dynamic loader for `scripts/check_architecture.py` (looks for `research/video_action_set/scripts/check_architecture.py`, which does not exist; the real file is at repo-root `scripts/check_architecture.py`).

### 3. Stale path (reorg regression): `test_video_verification_closure_kernel.py` (1 file)

Same `parents[1]` miscalculation as above: `REPO_ROOT` resolves to `research/video_action_set` instead of the repo root, so `SCRIPTS_ROOT / "check_architecture.py"` points at a directory that doesn't exist (`research/video_action_set/scripts/` is not present at all). This is an artifact of the test file being moved one level deeper (into a `tests/` subdirectory) during the split without updating the `parents[N]` arithmetic. Classification: **stale import/path**.

### 4. Marker-registration gap: 3 files use an unregistered `research` marker

`test_video_provider_measurement_kernel.py`, `test_video_provider_measurement_real.py`, `test_video_split_progress_kernel.py` all import cleanly (none touch the broken example stub) but declare `pytestmark = pytest.mark.research`. Root `pyproject.toml`'s `[tool.pytest.ini_options]` and `tests/conftest.py` register only `integration` and `slow`; `research` is registered nowhere in the repository, and root `addopts = "--strict-markers"` turns any unregistered marker into a hard collection error (`'research' not found in \`markers\` configuration option`).

Classification: this is **not** an obsolete test in the behavioral sense - it is pure marker-registration config drift. Closest bucket in the required taxonomy is **obsolete test** (configuration no longer matches usage), but the underlying test logic is intact and would run immediately once `research` is added to the root marker list.

## The 9 files that DID collect

All nine classify as **research benchmark** or **research evidence** - none are packages/core, packages/video, or packages/vision production-ownership candidates:

| file | classification |
|---|---|
| test_arcade_shooter_baseline_comparison.py | research benchmark (**hidden runtime regression**, see below) |
| test_arcade_shooter_example.py | research benchmark (**hidden runtime regression**, see below) |
| test_arcade_shooter_exhaustive.py | research benchmark (**hidden runtime regression**, see below); incidentally exercises packages/core bundle/policy-lookup round-trip across the full 7x8x2 state space |
| test_arcade_visual_address_benchmark.py | research benchmark |
| test_arcade_visual_local_baseline_postanalysis.py | research evidence |
| test_arcade_visual_local_baseline_showdown.py | research benchmark (already correctly tagged `@pytest.mark.slow`, a registered marker) |
| test_arcade_visual_registered_calibration_v2.py | research benchmark (`@pytest.mark.slow`) |
| test_visual_local_evidence_benchmark.py | research benchmark |
| test_visual_system_b.py | research evidence |

**Hidden runtime regression invisible to `--collect-only`:** three of these nine files (`test_arcade_shooter_baseline_comparison.py`, `test_arcade_shooter_example.py`, `test_arcade_shooter_exhaustive.py`) dynamically `importlib`-load `examples/arcade_shooter_policy.py` *inside* each test function rather than at module import time, so they collect successfully - but every test body then calls `demo.ACTIONS`, `demo.TinyArcadeShooter`, `demo.compile_policy_artifact`, `demo.state_row_id`, or `demo._action_values`, none of which exist in the rewritten 32-line stub. These will raise `AttributeError` the moment they actually execute. This is the same root cause as section 1, but it does not show up in a collection-only pass - actually running these three files is required to observe it.

## Split candidates (explicitly requested deep-dive files)

### `test_video_policy.py` - split recommended

10 of 11 test functions are **generic production invariants** of `zeromodel.video.video_policy.VideoPolicyReader` (impossible-transition rejection, temporal-gap bookkeeping, staleness horizon, frame-reorder detection, trace-id determinism, manifest tamper/mismatch detection, independent-evidence requirement) - arcade frames are only a convenience fixture, and every one of these properties would hold for any policy/config. Only `test_exact_canonical_video_reproduces_symbolic_rows_actions_and_trace` is a genuine **full arcade closed-world proof** (whole-universe symbolic-trace reproduction).

**Recommendation:** move the 10 generic tests to `packages/video/tests/` as a dedicated `VideoPolicyReader` unit suite (replacing the `_reader()` fixture's dependency on the broken `examples.arcade_shooter_policy` import with a minimal self-contained fixture policy, which also fixes their ImportError). Keep only the one closed-world proof in `research/`.

### `test_video_benchmark_facade.py` - split recommended, but not along the generic/closed-world axis

On close reading, **none** of this file's 5 test functions touch arcade gameplay semantics at all. All 5 are **repository architecture/tooling** tests: facade re-export/aliasing correctness, an AST check that the facade contains no real function bodies, monkeypatch-seam preservation through delegation, and two parametrized checks that dynamically load `scripts/check_architecture.py` to assert forbidden/allowed import edges for `zeromodel.domains.video_action_set.*`.

**Recommendation:** fix the three stacked bugs (typo, missing `artifact_io` import, `parents[1]` miscalculation), then consider relocating this file out of the arcade-specific research tree entirely, since it has no gameplay dependency - and merge its `check_architecture.py` layering tests with the next file's, since they test the identical checker against an overlapping edge set (likely duplicate coverage).

### `test_video_verification_closure_kernel.py` - split recommended, same finding

Also **zero** arcade-closed-world content. Mixes (a) generic unit tests of `research/video_action_set/verification.py`'s pure closure/summary logic against synthetic dict fixtures (no arcade content whatsoever), with (b) the same `scripts/check_architecture.py` layering tests found in `test_video_benchmark_facade.py`, testing an overlapping set of forbidden edges.

**Recommendation:** split into (1) a focused `test_verification_closure.py` covering just the synthetic closure/summary logic, and (2) merge its architecture-layering tests with `test_video_benchmark_facade.py`'s into one canonical `test_video_action_set_architecture_layering.py` - this also lets the shared `parents[1]` REPO_ROOT bug be fixed once instead of twice.

## Summary counts

| category | count |
|---|---|
| Total files inventoried | 30 |
| Tests collected | 25 (across 9 files) |
| Collection errors | 21 (across 21 files) |
| - stale import (examples/arcade_shooter_policy.py rewrite) | 16 |
| - syntax error | 1 (plus 2 latent bugs uncovered underneath) |
| - stale import/path (parents[1] reorg regression) | 1 |
| - marker-registration gap (obsolete-config bucket) | 3 |
| - missing external data / missing credentials / actual research regression | 0 |
| Collecting files with a hidden runtime-only regression | 3 |
| Files recommended for splitting | 3 (all three named in the audit brief) |

No repairs were made. No test was moved, skipped, or altered.
