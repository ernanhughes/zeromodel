# Stage 2D — Provider Evaluation & Policy-Impact Verification RMDTO Aggregate

**Baseline SHA (start of this stage):** `0ce62ca5af1f9d66c1efcb2dce00dcd384b54fb2` (branch `main`, working tree clean at start).
**Branch:** `stage-2d-provider-evaluation-aggregate`.
**Objective:** turn provider evaluation for the video action-set domain into a first-class, immutable, database-backed RMDTO aggregate that keeps *observation correctness* (exact state) and *application-behaviour correctness* (policy action) as separate, queryable evidence, per
[docs/architecture/provider-evaluation-rmdto.md](../architecture/provider-evaluation-rmdto.md). This stage does not build a model tuner - it builds the evidence substrate.

## What was built

DTOs (`packages/video/src/zeromodel/video/domains/video_action_set/`):
- `provider_evaluation_configuration_dto.py` - `ProviderConfigurationDTO` (content-derived identity, recursive secret-key rejection in `inference_options`/`metadata`).
- `provider_evaluation_case_dto.py` - `ProviderEvaluationCaseDTO`, `ProviderEvaluationCaseContext`, `ProviderResponseEvidence`, the four `CASE_OUTCOME_*` classifications, and the decision-trace validation that reuses `VPMPolicyLookup.PolicyLookupDecision` instead of re-declaring a second policy-decision shape.
- `provider_evaluation_summary_dto.py` - `ProviderEvaluationSummaryDTO`, with `from_cases(...)` as the one true builder and nearest-rank median/p95 over integer microseconds.
- `provider_evaluation_run_dto.py` - `ProviderEvaluationRunDTO` (aggregate root) and `MaterializedProviderEvaluationRunDTO` (where every aggregate-closure invariant lives), plus `build_provider_evaluation_run(...)`.
- `provider_evaluation_common.py` - shared primitives (`nonempty_str`, `nonneg_int`, `decision_payload`, etc.).
- `provider_evaluation_dto.py` - thin public façade re-exporting the above (mirrors `observation_dto.py`'s role for the observation aggregate).
- `provider_evaluation_service.py` - `ProviderEvaluationService`, a thin pass-through to `VideoActionSetStore`.

Store surface: `store.py` gains exactly five methods (`save_provider_evaluation_run`, `get_provider_evaluation_run`, `get_materialized_provider_evaluation_run`, `list_provider_evaluation_runs`, `list_provider_evaluation_cases`) plus five new conflict messages/`raise_*` helpers.

In-memory Store: `stores/provider_evaluation_memory.py` (`_ProviderEvaluationMemoryStoreMixin`, mixed into `InMemoryVideoActionSetStore`).

SQLAlchemy persistence:
- `db/orm/provider_evaluation.py` - `ProviderEvaluationConfigurationORM`, `ProviderEvaluationRunORM` (denormalized summary counts + one `summary_json` column - no separate summary table), `ProviderEvaluationCaseORM` (FK to `video_action_set_observation.frame_id`, `UniqueConstraint(run_id, case_ordinal)`).
- `db/stores/provider_evaluation.py` - DTO<->ORM mapping helpers, SQL predicate builders, and `ProviderEvaluationSqlStoreMixin` (mixed into `SqlAlchemyVideoActionSetStore` in `db/stores/video_action_set.py`).
- `db/session.py` - the three new ORM classes registered in `create_schema()`.

Engine/Facade/Runtime: `engine.py`/`facade.py` mirror the five Service methods 1:1; `runtime.py` wires `ProviderEvaluationService` into `build_runtime()`. SQLite composition continues unchanged through `zeromodel-sqlalchemy`'s existing `build_sqlite_runtime`.

Report adapter (composition seam, outside both packages): `examples/provider_evaluation_report_adapter.py` - `ProviderEvaluationReportAdapter` (implements `ReportAdapter[MaterializedProviderEvaluationRunDTO]`) and `compile_provider_evaluation_report(...)`, calling the existing `zeromodel.artifacts.report_compiler.compile_report` (no reimplementation of report persistence or closure validation).

Example refactor: `examples/local_model_zero_arcade_test.py` now materializes every rendered frame as an `ObservationDTO` (via a deterministic 16x28 grayscale thumbnail satisfying the existing pixel-materialization contract, while the full-resolution PNG remains the actual frame sent to the provider), builds a `ProviderEvaluationCaseDTO` per case, saves the complete run atomically, reloads and asserts closure, and writes `cases.jsonl`/`summary.json` from Store-returned DTOs. New flags: `--store {memory,sqlite}` (default `memory`), `--sqlite-path`, `--compile-report`.

## Key design decisions and deviations from the brief

- **No public `save_provider_configuration`/`get_provider_configuration`.** `ProviderConfigurationDTO` is embedded in the run and deduplicated internally by both Stores (same idempotent-conflict idiom as `save_matrix_blob`) - keeps the Store surface at exactly five methods, per "do not add speculative query methods without a demonstrated use."
- **A case belongs to exactly one run** (not content-addressed/shared like matrix blobs or configurations): a `case_id` colliding with an existing case - even with identical content - is always a conflict, both in-memory and in SQL. Verified by dedicated adversarial tests in both Stores.
- **No separate `provider_evaluation_summaries` SQL table.** A summary has no independent identity/lifecycle apart from its one run; it's stored as denormalized count columns (for query) plus one `summary_json` column (for exact reconstruction), mirroring how `EpisodeCountsDTO` is JSON-embedded rather than tabled.
- **`missing_value_semantics="error"` (dense matrix), not `"absent"`,** in the report adapter - `compile_report`'s sparse path isn't implemented yet (raises "no sparse VPM representation yet"). Inapplicable per-case dimensions get `raw_value=0.0` with `importance=0.0` (an explicit "do not weight this cell" flag) rather than a fabricated match/mismatch.
- **`ProviderEvaluationCaseDTO.build(...)` takes a `context: ProviderEvaluationCaseContext` and an `evidence: ProviderResponseEvidence` bundle** rather than ~16 flat keyword arguments - both to stay under the repository's 10-parameter hard limit and because those fields are genuinely one semantic unit each (the run-scoped identity every case shares; the secondary provider-response evidence for one case).
- **`ProviderConfigurationDTO.build(...)` omits `provider_version`/`runtime_version`/`context_length`** (no current caller uses them); they remain settable via `from_dict` directly if a future caller needs them.
- **VPM policy `artifact_id` is a bare hex digest** (Core's own convention), while every identity field in this aggregate uses the `sha256:`-prefixed convention used throughout `video_action_set`. The example normalizes at the one boundary crossing (`_decision_payload` helper), documented inline.

## Tests

61 new tests, all passing:
- `tests/test_video_provider_evaluation_rmdto.py` (50) - DTO construction/validation/digest/round-trip, summary math (exact fixture, imperfect fixture matching the brief's observed 3/4/1/7 shape, deterministic median/p95, reordering, empty case list, fabricated-count rejection), aggregate-closure adversarial tests (cases from another run, unrelated summary, mismatched policy artifact, non-contiguous ordinals, duplicate case id, missing case), in-memory Store tests (save/load, idempotent save, unknown observation, cross-run case-sharing rejection, stable filtered listing).
- `tests/test_video_provider_evaluation_sql_store.py` (11, `pytestmark = pytest.mark.integration`) - schema creation, round trip, idempotent save, rollback-on-failure (no orphan rows across all three tables), cross-run duplicate-case rejection, FK enforcement on `frame_id`, SQL-predicate filtering, tampered-row digest rejection (run counts, case outcome), in-memory/SQL parity, SQLite runtime composition.

Fixture A (exact) and Fixture B (3 exact / 4 action-equivalent / 1 action-changing / 7 action-correct, matching the brief's observed unlabelled-fixture shape) are both exercised directly against the DTOs/summary and end-to-end through the example (`--backend fake`).

## Validation run this session

Environment note: this machine's default Python interpreter is a **shared global interpreter** used by unrelated projects (`opencv-python`, `transformers`, `writer-runtime`). An initial `pip install -r requirements-dev.txt` into that shared interpreter downgraded `numpy` 2.2.3 -> 1.26.4 and reported dependency conflicts with those unrelated projects; the numpy downgrade was reverted immediately (`pip install numpy==2.2.3`) and all further validation ran inside a disposable, repo-local virtualenv (`python -m venv`, deleted after use, never committed) so the shared interpreter was not depended on further.

```
python -m pytest tests/test_video_provider_evaluation_rmdto.py tests/test_video_provider_evaluation_sql_store.py --run-integration -q
# 61 passed

python -m mypy packages/core/src packages/analysis/src packages/observation/src \
  packages/vision/src/zeromodel/vision/visual.py packages/vision/src/zeromodel/vision/visual_policy.py \
  packages/vision/src/zeromodel/vision/__init__.py packages/video/src packages/sqlalchemy/src \
  packages/artifacts/src packages/trust/src packages/navigation/src
# Success: no issues found in 152 source files

python scripts/check_package_boundaries.py    # Package boundary check passed: 152 production modules
python scripts/check_architecture.py          # Architecture check: passed (152 production modules inspected)
python scripts/run_fast_tests.py              # 1075 passed, 1 skipped, 82 deselected, 0 failed (111.41s / 120s budget)
python scripts/check_quality.py               # Quality checks passed
python -m ruff check .                        # All checks passed (repo-wide, read-only)
```

`python -m ruff format .` was **not** applied repo-wide: an initial run reformatted 143 pre-existing files unrelated to this change (formatting drift already present on `main`, unrelated to this stage). Those were reverted; `ruff format` was instead applied only to the files this stage actually touches.

Manual example runs (`--backend fake`, both exact-fixture smoke states):
- `--store memory` (default): 8/8 exact, 8/8 accepted, reload-verified, exit 0.
- `--store sqlite --sqlite-path <path>`: same result via `zeromodel-sqlalchemy` runtime.
- `--compile-report`: compiles, persists, and reloads the VPM report via `load_compiled_report_aggregate` with full closure validation; `report.json` written.

## Quality-gate ceiling adjustments

Two pre-existing "legacy exception" line-count ceilings in `quality-baseline.toml` were raised by the minimum amount needed to mix the new Store methods in (the methods themselves live in new, separate modules - `db/stores/provider_evaluation.py` and `stores/provider_evaluation_memory.py` - not grown inline):
- `packages/sqlalchemy/.../db/stores/video_action_set.py`: 1333 -> 1338 lines (+5: one new import statement, one wrapped class declaration).
- `packages/video/.../stores/video_action_set_memory.py`: 703 -> 719 lines (+16: same pattern).

This follows the existing precedent of `final_access_service.py`'s ceiling being raised with a documented reason for the pre-audit stabilization TOCTOU fix.

## Remaining limitations / recommended next steps

- The auto-generated package-inventory family (`docs/architecture/package-inventory-1.0.13.md`, `package-module-map-1.0.13.csv`, `package-public-api-1.0.13.csv`, etc.) was **not** regenerated - those are versioned release-process snapshots generated by `scripts/analyze_package_inventory.py`, and regenerating them is a release-cut concern, not a per-PR one. Module/API counts there will be stale by the seven new modules added here until the next release cut regenerates them.
- External Ollama execution remains manual/external, as required - not part of the fast suite.
- Recommended next practical experiment: re-run the local-model arcade example against a real local vision model (both labelled and unlabelled smoke fixtures) with `--store sqlite --compile-report`, and compare the persisted `action_changing_count`/`action_equivalent_count` split against the ad hoc numbers previously recorded in `docs/results/local-model-zero-arcade-smoke-v1/README.md`, to confirm the new aggregate reproduces that prior evidence exactly before treating it as the system of record.
