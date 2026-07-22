# Post-split remediation — Stage 3: `zeromodel-navigation` finite identified hierarchy

**Baseline SHA (start of this stage):** `c1ce710db50655a6082567fd3f376c3134095ea2` (branch `main`, working tree clean at start — same preflight check as Stage 2, see [post-split-stage-2-trust-validation.md](post-split-stage-2-trust-validation.md)).
**Final SHA / working-tree state:** uncommitted at the time of writing — nothing committed, tagged, published, or pushed in this stage.
**Objective:** compile and traverse a finite, deterministic hierarchy over identified artifacts, explicitly *not* search — no similarity/relevance definitions, no learned or heuristic routing.

## Two hierarchy concepts kept separate

- `zeromodel.analysis.hierarchy.build_pyramid()` (existing, untouched) — reduces **one** VPM field into coarser levels of itself (an intra-artifact operation).
- `zeromodel.navigation` (new) — a corpus hierarchy over **many** identified artifacts: root tile → internal navigation tiles → leaf artifact bindings. No file in `packages/analysis` was modified.

## Architecture built

- **New distribution:** `zeromodel-navigation`, namespace `zeromodel.navigation`, `packages/navigation/`.
- **Dependency rule enforced:** `navigation → core + artifacts` only. Navigation does not import `zeromodel.trust` anywhere (enforced by a dedicated import-isolation test and by `check_package_boundaries.py`).
- **No second content-addressed repository:** every `NavigationTileDTO` / `LeafBindingDTO` is persisted as canonical bytes through the Artifacts package's `ArtifactStore`/`ArtifactResolver` protocol ([storage.py](../../packages/navigation/src/zeromodel/navigation/storage.py)). A tile's identity payload bytes are handed to the store as `canonical_bytes` so the store's own content digest equals the tile's declared `tile_id` exactly; the full DTO content is kept as the store's `manifest` for resolution back into a DTO. Navigation owns the DTO semantics; Artifacts owns storage and identity.

### Files

- `packages/navigation/pyproject.toml`, `README.md` (includes the Trust-composition integration-seam example, without importing Trust).
- `dto.py` — 12 DTOs: `HierarchyManifestDTO`, `NavigationTileDTO`, `TileCoverageDTO`, `TilePointerDTO`, `LeafBindingDTO`, `HierarchyCompilerSpecDTO`, `TraversalRuleDescriptorDTO`, `TraversalRequestDTO`, `TraversalStepDTO`, `TraversalResultDTO`, `TraversalFailureDTO`, `TraversalReceiptDTO`. `tile_id`, `leaf_id`, `hierarchy_id`, `receipt_id` are all self-validating content digests over their own canonical payload — reordering a tile's children changes its `tile_id`; changing any bound field (root ref, source artifact set, compiler identity/version, partition parameters, child ordering, tie rule, failure rule, leaf semantics, navigation rule contract) changes the `hierarchy_id`.
- `compiler.py` — `compile_hierarchy` (deterministic, order-preserving grouping compiler; rejects corpus-artifact-kind mismatches; calls `validate_hierarchy` internally before returning, so the compiler never hands back a hierarchy that would fail its own closure check) and `validate_hierarchy` (exhaustive, finite structural walk — root resolves, every child/leaf reference resolves, no self-reference, cycle detection via a path stack, bounded depth against the hierarchy's declared `max_depth`, no duplicate children, declared leaf semantics and corpus-artifact-kind contract enforced on every leaf). This walk is exhaustive verification of an already-built finite structure, not a similarity search.
- `rules.py` — the `TraversalRule` protocol (`descriptor()`, `select_child(request, tile, children)`) plus two reference, explicitly non-search implementations: `FixedKeySelectorRule` (exact match against a declared request attribute) and `DeclaredPriorityRule` (fixed index/target-range routing). A later Search package can implement the same protocol with similarity-driven rules without Navigation changing.
- `traversal.py` — `traverse` (walks from the root, recording every step — including ties and failures — as `TraversalStepDTO` data) and `replay_traversal` (re-executes a `TraversalReceiptDTO`'s recorded request against the same hierarchy/rule and returns a fresh, comparable result; rejects a receipt/manifest `hierarchy_id` mismatch).
- `__init__.py` — the public surface is restricted to exactly the 12 names the stage brief specified: `HierarchyManifestDTO`, `NavigationTileDTO`, `HierarchyCompilerSpecDTO`, `TraversalRule`, `TraversalRequestDTO`, `TraversalStepDTO`, `TraversalResultDTO`, `TraversalReceiptDTO`, `compile_hierarchy`, `validate_hierarchy`, `traverse`, `replay_traversal`. Supporting DTOs (`TileCoverageDTO`, `TilePointerDTO`, `LeafBindingDTO`, `TraversalRuleDescriptorDTO`, `TraversalFailureDTO`) remain reachable from `zeromodel.navigation.dto` for tests/advanced composition but are deliberately excluded from `__all__`.

## A structural property worth recording

Content-addressed identity makes genuine self-reference or cycles *structurally infeasible* to construct honestly (a tile's id is a hash of its own children's ids, so a cycle would require solving a hash fixed point). The two closure tests for these cases (`test_self_referencing_tile_fails_closure`, `test_cyclic_hierarchy_fails_closure`) therefore exercise the defensive guard directly via `monkeypatch` against a simulated corrupted store, rather than via the real compiler — this is noted in-line in both tests rather than left implicit.

## Tests

`packages/navigation/tests/` — 21 tests, all passing:

- `test_navigation_api_isolation.py` (3): restricted public API set, import isolation (core + artifacts load, `zeromodel.trust` never loads), wheel-content check.
- `test_compile_and_closure.py` (11): identical input+spec → identical hierarchy identity, changed child order → different identity, changed spec rule → different identity, incompatible corpus artifact fails compilation, max depth enforced at compile time, root and every reference resolves, missing leaf reference fails closure, duplicate child rejected at tile construction, self-referencing tile fails closure, cyclic hierarchy fails closure, Navigation owns no independent persistence (a hierarchy compiled into one store does not resolve against a different, unrelated store).
- `test_traversal.py` (7): reaches expected leaf (single- and multi-level), deterministic tie resolution, failure represented as data (not an exception), max depth enforced during traversal, traversal receipt replays to the same path, replay rejects a mismatched hierarchy.

## Governance integration

Same wiring as Stage 2 — see [post-split-stage-2-trust-validation.md](post-split-stage-2-trust-validation.md)'s governance section; Navigation additionally has its own `.github/workflows/navigation-package.yml`.

## Claims boundary (as documented in `packages/navigation/README.md`)

Supported: compiling and deterministically traversing a finite, identified hierarchy with complete artifact resolution and a replayable trace.
Explicitly not claimed: planet-scale hierarchies, infinite in-memory capacity, logarithmic-time guarantees, semantic search, nearest-neighbour retrieval, "40-hop world navigation," or storage-independent performance.

## Validation run this session

```
python -m pytest -q packages/navigation/tests       # 21 passed
python -m mypy packages/navigation/src               # Success: no issues found
python -m ruff check packages/navigation/src packages/navigation/tests    # All checks passed
python -m ruff format --check packages/navigation/src packages/navigation/tests  # passed (after one format pass)
python scripts/check_quality.py                       # Quality checks passed (all 9 packages)
python scripts/run_fast_tests.py                      # see combined report below
python scripts/validate_release_candidate.py          # Release candidate validation passed
```

## Explicitly not run without further authorization

- Large hierarchy builds (thousands of artifacts), traversal benchmarks, or any `@pytest.mark.slow` scenario for Navigation — none exist yet, and none were added, since the stage brief prohibits running slow/benchmark tests without explicit authorization.
- Building/publishing to TestPyPI or PyPI — not triggered.
- No git commit, push, or tag was made for this stage's changes.

---

# Combined governance wiring (Stage 2 + Stage 3)

Both new packages, plus the implicit Stage 1 `zeromodel-artifacts`, were wired into every governance surface in the same pass:

- **`requirements-dev.txt`**: `-e ./packages/artifacts`, `-e ./packages/trust`, `-e ./packages/navigation`, `cryptography>=41`.
- **Root `pyproject.toml`**: `pythonpath` and `mypy_path` gained all three `src` roots.
- **`package-boundaries.toml`**: three new `[packages.*]` sections — `artifacts` (`depends_on = ["core"]`), `trust` (`depends_on = ["core", "artifacts"]`), `navigation` (`depends_on = ["core", "artifacts"]`). No package outside these three declares a dependency on `trust` or `navigation`.
- **`scripts/run_fast_tests.py`**: `TEST_ROOTS` gained `packages/artifacts/tests`, `packages/trust/tests`, `packages/navigation/tests` (now 11 roots total).
- **`scripts/check_quality.py`**: `FORMAT_LINT_PATHS`, `TYPING_PATHS`, `QUALITY_LIMIT_PATHS`, and both the package-boundaries/architecture `run_step` display lists gained all three packages' `src`/`tests` paths.
- **`scripts/validate_release_candidate.py`**: `PACKAGES` gained `artifacts`, `trust`, `navigation` entries with correct `requires`/`depends_on`/`wheel_stem`.
- **CI**: three new workflows (`artifacts-package.yml`, `trust-package.yml`, `navigation-package.yml`) mirroring the existing per-package pattern (`sqlalchemy-package.yml`); `python.yml`'s nine-package build/twine loop and `package-integration.yml`'s job label updated from "six-package" to reflect the real count.
- **`.vscode/settings.json`**: `python.analysis.extraPaths` and `python.testing.pytestArgs` extended to all three new packages.
- **Regression tests updated** (each was an intentional "cannot silently shrink" exact-list/exact-set guard that correctly failed until updated, confirming the guards work): `tests/test_developer_tooling_consistency.py`, `tests/test_fast_suite_completeness.py`, `tests/test_package_boundaries.py`, `tests/test_release_package_metadata.py`, `tests/test_workspace_ci_invariants.py`, `tests/test_public_api_manifest.py`.

## Full fast-suite result (final run, this session)

```
Collected: 950, deselected: 82, passed: 867, failed: 0, skipped: 1
```

Four consecutive runs were taken to check runtime stability after adding 64 new package-local tests:

| run | runtime |
|---|---|
| 1 (first run after installing 3 new packages/rebuilding wheels) | 107.84s |
| 2 | 87.58s |
| 3 | 79.88s |
| 4 | 90.73s |

The first run's elevated time is consistent with cold bytecode-cache/import overhead immediately after installing three new editable packages; the following three consecutive runs give a slowest of 90.73s and a median of 87.58s — both within the acceptance targets Stage A2.1 established (slowest < 100s, median < 90s), against the 120s budget.

## Full quality gate result (final run, this session)

```
python scripts/check_quality.py
...
Quality checks passed
```

Ruff format, ruff lint, mypy, package boundaries, architecture rules, and code-quality limits (function/module/class size, nesting, parameter count) all pass across all nine packages with zero legacy exceptions added for the three new packages. Two real defects were caught and fixed during this stage's own quality pass before commit-readiness: three functions (`verify_artifact_for_scope`, `compile_hierarchy`, `traverse`) initially exceeded the 100-line hard function-length limit and were refactored into single-concern helpers (no behavior change — full test suite re-verified green after each refactor); a `mypy` type mismatch in `compute_authorization_id`'s internal draft-payload helper was resolved by replacing the ad hoc stand-in dataclass with a shared, explicitly-typed field-based payload builder.

## Full release-candidate validation result (final run, this session)

```
python scripts/validate_release_candidate.py
...
docs/architecture/package-public-api-1.0.13.csv: 258 public symbols across 9 distributions
Release candidate validation passed
```

This rebuilt all nine wheels, installed them into a clean venv, ran `pip check` (no broken requirements), and regenerated the public API manifest — required because the manifest is a checked-in CSV snapshot, not something several fast-suite tests regenerate themselves.

## Explicitly not run without further authorization (repository-wide)

- `@pytest.mark.slow` / `@pytest.mark.external` / `@pytest.mark.research` tests anywhere in the repo (unaffected by this stage, still excluded from the fast suite by marker as established in Stage A2).
- `python -m twine upload` / any TestPyPI or PyPI publish step.
- Any git commit, push, tag, or branch operation — the working tree remains uncommitted pending explicit instruction.
