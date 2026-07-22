# Post-split Stage A2 — public API manifest validation

## Before

`scripts/validate_release_candidate.py`'s `write_public_exports()` wrote exactly one placeholder row per distribution, with the literal string `"__all__"` in the `exported_symbol` column - never a real symbol name. `docs/architecture/package-public-api-1.0.13.csv` had 6 data rows total, covering roughly 2.8% of the real public surface.

## After

`write_public_exports()` now:

1. Runs a probe script inside the clean venv built by `install_and_probe()` (all six wheels installed, nothing importable from the checkout) - an isolated validation environment, not the developer's editable-source environment.
2. Imports each package's declared namespace module and reads its real `__all__`.
3. For every symbol: resolves `object_kind` (Class / Function / Constant) via `inspect`, and `source_module` via `obj.__module__` (falling back to the namespace itself for values with no `__module__`, e.g. plain constants).
4. Validates, failing the whole run on any violation:
   - no duplicate symbol within one namespace's `__all__`;
   - no symbol name starting with `_`;
   - every symbol's `source_module` belongs to that package's own namespace or one of its **declared** dependencies (read from the same `PACKAGES["depends_on"]` structure used for version validation) - catches an export whose implementation lives in an undeclared sibling package;
   - the manifest has more than one row per distribution (rejects a regression back to placeholder rows).
5. Writes one row per real symbol, sorted by `(distribution, exported_symbol)` for determinism.

Required columns are exactly: `distribution, namespace, exported_symbol, owning_module, object_kind, source_module, is_reexport`.

## Result

Regenerated `docs/architecture/package-public-api-1.0.13.csv`: **211 rows**, matching the independently-computed real `__all__` totals across all six packages exactly:

| distribution | real `__all__` count | manifest rows |
|---|---|---|
| zeromodel (core) | 52 | 52 |
| zeromodel-analysis | 87 | 87 |
| zeromodel-observation | 12 | 12 |
| zeromodel-vision | 18 | 18 |
| zeromodel-video | 32 | 32 |
| zeromodel-sqlalchemy | 10 | 10 |
| **Total** | **211** | **211** |

Verified:
- **Determinism:** regenerating the manifest twice in a row (same clean venv) produced byte-identical output (`diff` empty). A `@pytest.mark.slow` regression test (`tests/test_public_api_manifest.py::test_manifest_generation_is_byte_identical_across_runs`) rebuilds the full six-package venv and asserts this directly.
- **Completeness/no-undeclared-export:** `tests/test_public_api_manifest.py` (12 fast tests) confirms every real `__all__` symbol appears in the manifest exactly once, no manifest row references a non-existent attribute, and every row's `source_module` stays inside that distribution's declared dependency closure (i.e., no package's public API secretly re-exports an undeclared sibling's implementation).
- **Not a placeholder:** the manifest has 211 rows, not 6; no row has `exported_symbol == "__all__"`.

No package's `__all__` list itself was changed. This is a tooling fix only - the manifest now accurately describes an unchanged public API.
