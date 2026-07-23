# Provider Evaluation & Policy-Impact Verification RMDTO Aggregate

Stage 2D of the video action-set RMDTO architecture. Turns one external
perception-provider evaluation run into a first-class, immutable,
database-backed aggregate, preserving the distinction between:

- **observation correctness** - did the provider recover the exact state; and
- **application-behaviour correctness** - did the compiled policy still choose
  the expected action.

A provider can predict the wrong exact state while the compiled policy action
still matches (`action_equivalent`); a provider error that changes the
compiled policy action (`action_changing`) is a materially different,
policy-boundary-crossing failure. This aggregate keeps those two measurements
as separate, queryable evidence rather than collapsing them into one accuracy
number, which is exactly what the ad hoc `cases.jsonl`/`summary.json` output
of the local-model arcade example could not do.

## Why this exists

The local-model arcade example
(`examples/local_model_zero_arcade_test.py`) established a real provider
boundary: a rendered PNG frame goes to a local vision-language model, which
returns a small bounded text description; a deterministic parser turns that
into a typed prediction; `VPMPolicyLookup` addresses an independently
compiled policy table and returns an action. The unlabelled fixture showed
exact-state accuracy (3/8) and policy-action accuracy (7/8) diverge sharply,
and one case crossed a policy boundary entirely. That evidence used to live
only in a JSON file this stage makes it a queryable, closure-validated
database aggregate.

## Provider isolation

The external perception provider (`Provider.predict` in
`examples/local_model_zero_arcade_test.py`) receives only observable input -
`(image: bytes, render_mode: str)` - and returns a bare, unlabelled reply.
Ground truth (`ArcadeState`, expected row, expected action) never crosses
that call boundary; the harness computes `exact_state_match`/`action_match`/
`outcome` afterward, entirely on its own side, by comparing the provider's
reply against truth it already held.

The test double reflects that boundary rather than working around it:
`ScriptedProvider` looks up a pre-scripted reply by the image's content
digest (`sha256:` of the PNG bytes) computed at `predict()` time - it does
not see or accept a truth argument. The function that *builds* the
digest-to-reply table (`_build_scripted_replies`) runs before any `predict()`
call and is allowed to know truth (it is the fixture, not the provider under
test); the provider call boundary itself is not. This is proven with a
runtime-behavioral test (a spy subclass recording every argument an
end-to-end `run()` actually passes to `predict()`) rather than relying on
static signature inspection alone, though a lightweight signature assertion
is kept as a secondary guard.

## Aggregate ownership and dependency direction

Same layering as every other `video_action_set` aggregate:

```
Runtime -> Facade -> Engine -> Service -> Store -> ORM -> Database
```

- `ProviderEvaluationService` is a thin pass-through to `VideoActionSetStore`
  (`provider_evaluation_service.py`) - it does not invoke a visual model,
  render or mutate PNGs, or contain aggregate logic beyond what the DTOs
  already enforce.
- `VideoActionSetEngine`/`VideoActionSetFacade` mirror the Service's five
  methods 1:1, zero logic, exactly like every other capability in this
  domain.
- `InMemoryVideoActionSetStore` and `SqlAlchemyVideoActionSetStore` both
  implement the same `VideoActionSetStore` protocol surface
  (`store.py`); the SQL mixin lives in `db/stores/video_action_set.py`
  alongside the existing observation mixin, and its DTO-to-ORM mapping
  helpers live in the companion `db/stores/provider_evaluation.py` module -
  the same file-per-concern split the observation aggregate already uses.
- The aggregate does not duplicate pixel payloads or transformation
  provenance. `ProviderEvaluationCaseDTO.frame_id` references an existing
  `ObservationDTO`/`MatrixBlob`/`ObservationOperationChainDTO` by identity;
  the operation chain already attached to that observation is the
  provenance of whatever visual intervention (rendering, transformation)
  produced the evaluated frame.

## DTOs

All in `packages/video/src/zeromodel/video/domains/video_action_set/provider_evaluation_dto.py`,
following the domain's established conventions: `@dataclass(frozen=True, slots=True)`,
canonical digests via `canonical_json.canonical_sha256`, exact-key
`from_dict`/`to_dict` round trips, `VPMValidationError` on every violation.

### `ProviderConfigurationDTO`

The provider/model/runtime configuration used for one run: `provider_kind`,
`model_name`/`model_digest`, `runtime_name`, `protocol_version`,
`prompt_digest`, optional `context_length`/`seed`, and canonical-JSON
`inference_options`/`metadata`. Identity (`provider_configuration_id`) is
content-derived; no wall-clock field participates. A recursive key-fragment
scan rejects `api_key`, `token`, `password`, `secret`, `bearer`,
`authorization`, and `credential`-shaped keys anywhere in
`inference_options`/`metadata` - secrets are structurally prevented from
entering this aggregate, not merely discouraged by convention.

### `ProviderEvaluationCaseDTO`

One evaluated observation. Reuses `VPMPolicyLookup.PolicyLookupDecision`
rather than re-declaring a second policy-decision shape:
`expected_decision_trace`/`predicted_decision_trace` store
`PolicyLookupDecision.to_dict()` verbatim, and `__post_init__` cross-checks
the trace's `artifact_id`/`row_id`/`action` against the case's own
`policy_artifact_id`/`expected_row_id` (or `predicted_row_id`)/`expected_action`
(or `predicted_action`) - proof that the row/action the case claims is
exactly the row/action the policy trace actually produced.

`exact_state_match`, `action_match`, `factor_matches`, and `outcome` are
stored fields, but `__post_init__` recomputes each from `expected_state`/
`predicted_state`/`expected_action`/`predicted_action`/`accepted` and raises
on any mismatch - the same "stored but recomputed-and-compared" idiom as
`ObservationDTO.action_known`. None of the four can be supplied dishonestly:
supplying a wrong value simply fails construction.

Outcome classification (`CASE_OUTCOME_*`):

```
rejected          - provider result was not admitted to policy lookup
exact             - accepted, exact state match, action match
action_equivalent - accepted, state not exact, action still matches
action_changing   - accepted, predicted action differs from expected
```

`action_changing` takes priority over `exact`/`action_equivalent` in the
derivation, so an internally inconsistent case (exact state but different
action - which should never happen for a deterministic policy) can never
silently report as `exact`.

A rejected case carries no predicted anything (`predicted_state`,
`predicted_row_id`, `predicted_action`, `predicted_decision_trace` are all
`None`); an accepted case carries all of them. Latency is stored in integer
microseconds, not float. Provider confidence is likewise stored as a
canonical scaled integer, `provider_confidence_basis_points` (`None` or an
`int` in `0..10000`) rather than an identity-bearing float - float
non-determinism must never leak into a content digest. A derived, non-stored
`provider_confidence` property (`float | None` in `0.0..1.0`) is recomputed
from the integer on every access for display/reporting convenience; it never
participates in `case_id`'s identity. `provider_raw_response_digest` is validated against
`provider_raw_response_text` when the text is retained (recomputed via a
domain-separated digest); when the text is intentionally discarded, only the
digest is kept as evidence.

`ProviderEvaluationCaseDTO.build(...)` is the ergonomic constructor: it
accepts a `PolicyLookupDecision` (or a plain mapping with the same shape)
directly for `expected_decision`/`predicted_decision`, derives every stored
field, computes the content digest, and constructs through `from_dict` so the
digest is correct from the first call (no placeholder-then-recompute step).

### `ProviderEvaluationSummaryDTO`

A deterministic, source-count-preserving summary: attempted/accepted/rejected
counts, exact/action_equivalent/action_changing/action_correct counts,
per-factor correct-vs-denominator counts (denominators counted only over
accepted cases - a factor absent from a rejected case is not "wrong", it is
inapplicable), rejection-reason counts, and latency
sample-count/min/max/total/median/p95.

Percentiles use the **nearest-rank method on integer microseconds**:
`rank(p) = ceil(p * n)` clamped to `[1, n]`; `value = sorted(values)[rank - 1]`.
Median uses `p=0.50`, p95 uses `p=0.95`. An empty case list is explicit and
allowed: every count is `0`, every latency field is `None`.

`ProviderEvaluationSummaryDTO.from_cases(cases)` is the one true builder - it
sorts by `case_ordinal` first (so reordered input normalizes to the same
summary) and is the only place these counts are computed from source
evidence. `__post_init__` only checks internal arithmetic consistency it can
see on its own (`attempted == accepted + rejected`, and so on) plus its own
digest; it cannot check reconciliation against the actual cases because it
does not hold them - that full reconciliation is a `MaterializedProviderEvaluationRunDTO`
concern, exactly mirroring how `EpisodeCountsDTO` doesn't validate against
episodes but `SealedSplitPlanDTO` does.

### `ProviderEvaluationRunDTO`

The aggregate root: `fixture_identity`, an embedded `ProviderConfigurationDTO`
(not just an id - mirrors how `SealedSplitPlanDTO` embeds full
`EpisodePlanDTO`s), `policy_artifact_id`, `case_mode`, `representation_mode`,
ordered unique `case_ids`, and the `ProviderEvaluationSummaryDTO`. It does not
hold the case list, so it only validates its own shape (digest, unique case
ids, sha256-shaped ids); it cannot check ordinal contiguity or summary
reconciliation by itself.

### `MaterializedProviderEvaluationRunDTO`

Run + ordered cases + summary. **Every aggregate-closure invariant lives
here**, in `__post_init__`:

1. case count matches `len(run.case_ids)`, and every case's `case_ordinal`
   equals its position (contiguous, zero-based);
2. `tuple(case.case_id for case in cases) == run.case_ids` (order + identity);
3. every case's `policy_artifact_id`/`provider_configuration_id` matches the
   run's;
4. `summary == run.summary` (proves the top-level field wasn't swapped for an
   unrelated-but-valid summary);
5. `summary == ProviderEvaluationSummaryDTO.from_cases(cases)` (full
   reconciliation against the actual cases).

Because of (4) and (5) together, it is impossible to combine a valid run,
cases from a different run, and an unrelated but individually valid summary -
individual validity is not aggregate closure.

`build_provider_evaluation_run(...)` (module function, not a classmethod - it
constructs and returns the materialized wrapper, not just the run) is the
builder: sorts cases, computes the summary via `from_cases`, builds the run,
and wraps everything in the closure-validated `MaterializedProviderEvaluationRunDTO`.

## Store semantics

`VideoActionSetStore` gains exactly five methods - no speculative extras:

```python
save_provider_evaluation_run(run) -> MaterializedProviderEvaluationRunDTO
get_provider_evaluation_run(run_id) -> ProviderEvaluationRunDTO | None
get_materialized_provider_evaluation_run(run_id) -> MaterializedProviderEvaluationRunDTO | None
list_provider_evaluation_runs(*, fixture_identity=, provider_kind=, model_digest=, policy_artifact_id=, case_mode=, representation_mode=)
list_provider_evaluation_cases(*, run_id=, outcome=, accepted=, exact_state_match=, action_match=, frame_id=)
```

No public `save_provider_configuration`/`get_provider_configuration` exists.
`ProviderConfigurationDTO` is embedded in the run and deduplicated internally
by both stores (keyed by `provider_configuration_id`, the same
idempotent-conflict idiom `save_matrix_blob` uses) - nothing outside a run
currently needs to address a configuration on its own, and the brief is
explicit that speculative query methods should not be added without a
demonstrated use.

**Evaluation cases are run-owned.** Unlike matrix blobs or provider
configurations - genuinely content-addressed, shared, reusable identities - a
`case_id` that already exists in the store (under any run) is always a
conflict on save, never a dedup opportunity. Note that `case_id` does not
itself encode `run_id`: two different runs could, in principle, produce a
byte-identical case (same ordinal, frame, policy, configuration, expected/
predicted state) and collide on the same `case_id`. `case_ordinal`
participating in the digest does not make ownership a structural guarantee
of the identity scheme by itself - it is enforced. Aggregate validation
(`MaterializedProviderEvaluationRunDTO.__post_init__`) binds the ordered
`case_id`s to the run, and both Store implementations reject reuse of an
already-persisted `case_id` by another run, treating any such collision as a
conflict rather than a legitimate replay.

`save_provider_evaluation_run` preflights, before any mutation: every case's
`frame_id` resolves to an existing observation; the provider configuration
either doesn't exist yet (insert) or matches exactly (reuse); every case's
`policy_artifact_id` matches the run's; no case id is already claimed. An
identical resave of an existing run is idempotent (returns the existing
aggregate); a conflicting payload under an existing `run_id` is rejected. The
SQL store runs all of this inside one transaction, so a failed case anywhere
in a batch rolls back the entire write - no orphan run/case/configuration
rows.

## SQL schema

Three tables, not four: `provider_evaluation_configurations`,
`provider_evaluation_runs`, `provider_evaluation_cases`. There is no separate
`provider_evaluation_summaries` table - a summary has no independent identity
or lifecycle apart from its one owning run, so it is stored as denormalized
count/latency columns on `ProviderEvaluationRunORM` (for SQL predicate
filtering) plus one `summary_json` column holding the full canonical payload
for exact reconstruction and digest reproof. This mirrors how
`EpisodeCountsDTO` is JSON-embedded in `EpisodePlanORM.payload_json` rather
than given its own table - a summary's counts are cheap to denormalize as
columns, but a fourth table would exist solely to be joined 1:1 with the run
that owns it.

Case rows carry real, indexed columns for everything the recommended filters
need (`run_id`, `frame_id`, `policy_artifact_id`, `provider_configuration_id`,
`accepted`, `exact_state_match`, `action_match`, `outcome`,
`expected_action`/`predicted_action`, `provider_latency_us`), a
`UniqueConstraint(run_id, case_ordinal)`, and a foreign key to
`video_action_set_observation.frame_id` - the mechanical proof that every case
references a real, persisted observation. Everything else (states, decision
traces, factor matches, provider metadata) is canonical JSON text, matching
the existing convention of "real columns for what must be queried or
constrained, canonical JSON for arbitrary structured evidence."

One implementation note for future SQL work in this domain: SQLAlchemy's
unit-of-work only topologically sorts INSERT order between mapped classes
that are connected by an ORM `relationship()` - raw `ForeignKey` columns
alone are not enough for it to infer that a run row must be inserted before
its case rows in the same flush. The SQL mixin here flushes after adding the
run and before adding cases for exactly this reason; anyone adding a new
child table with a raw foreign key to a freshly-inserted parent in the same
transaction needs the same explicit flush.

## Exact vs action-equivalent vs action-changing, restated

```
exact:             accepted, exact state match, action match
action_equivalent: accepted, state not exact, action still matches
action_changing:   accepted, predicted action differs from expected
rejected:          provider result was not admitted to policy lookup
```

This is the aggregate's central contribution: it is possible - and, per the
arcade example's own unlabelled fixture, common - for a provider to be wrong
about the exact world state while the compiled policy still produces the
correct action. Reporting only "exact-state accuracy" or only "action
accuracy" hides one of these two measurements; this aggregate always reports
both, plus the boundary-crossing count that neither one surfaces on its own.

## Report adapter (composition seam, outside both packages)

`examples/provider_evaluation_report_adapter.py` translates a
`MaterializedProviderEvaluationRunDTO` into the neutral `AdaptedReportDTO`
shape via the `ReportAdapter` protocol and calls the existing
`zeromodel.artifacts.report_compiler.compile_report` - it does not
reimplement report persistence, VPM persistence, or aggregate-closure
validation.

This lives in `examples/`, not in `packages/video` or `packages/artifacts`,
because `package-boundaries.toml` forbids either production package from
importing the other (`video` depends on `["core", "observation"]`;
`artifacts` depends on `["core"]`) and `scripts/check_package_boundaries.py`
only scans `packages/*/src` - `examples_root` is outside its reach entirely.
This is the same reason `writer_report_adapters_demo.py` composes its own
adapter outside `zeromodel.artifacts` for a different domain; concrete
`ReportAdapter`s are meant to live in the external application, never inside
the packages they bridge.

The current report compiler requires a dense matrix
(`missing_value_semantics="error"` is the only implemented path; `"absent"`
raises "no sparse VPM representation yet"). Inapplicable cells (a
predicted-state factor on a rejected case, or a factor key absent from that
case's `expected_state`) are represented by a placeholder `raw_value=0.0`
with `importance=0.0`. The zero is not measured negative evidence and must
not be interpreted independently of applicability or importance - these
cells are never "omitted", they are explicit placeholders. Applicable cells
always carry `importance=1.0` (`raw_value=1.0` for a measured true,
`raw_value=0.0` for a measured false - the same raw value as a placeholder,
distinguished only by `importance`). Each cell's `source_binding.attributes`
additionally tags `("applicable", "true"|"false")` (plus
`("placeholder", "true")` when inapplicable), so applicability can be read
directly off the source binding rather than inferred from
`raw_value`/`importance` alone. Subjects (cases) are ordered by consequence:
action-changing first, then rejections, then accepted-non-exact cases, then
exact cases, so the compiled report's row order concentrates the most
consequential failures in the inspection region.

## What remains external research

This stage does not:

- tune, fine-tune, or select models;
- search context sizes or runtime configurations automatically;
- implement an error-correcting decoder or prove Hamming-style correction;
- implement a general PNG transformation library or define the scientific
  meaning of any transformation;
- provide production Ollama (or any other) client code - `OllamaProvider`
  stays in `examples/`;
- escalate to larger models at runtime or calibrate confidence automatically;
- introduce a new package, database, or artifact store.

It is the evidence substrate: a persisted, closure-validated record of what a
provider actually returned, what the policy actually decided, and how those
two differ, that any of the above would need to consume.
