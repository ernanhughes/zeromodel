# Evidence Contract Compiler

- Implementation: [`examples/visual_transition_benchmark/compiler/`](../../examples/visual_transition_benchmark/compiler/) (`contracts.py`, `candidates.py`, `evaluate.py`, `compile.py`), [`compiler_adapters/`](../../examples/visual_transition_benchmark/compiler_adapters/) (`arcade.py`, `warehouse.py`), [`compiler_run.py`](../../examples/visual_transition_benchmark/compiler_run.py)
- Frozen result record: [`docs/results/evidence-contract-compiler-v1/`](../results/evidence-contract-compiler-v1/)
- Prior stages: [`visual-transition-debugging-benchmark.md`](visual-transition-debugging-benchmark.md), [`value-aware-transition-contracts.md`](value-aware-transition-contracts.md), [`cross-domain-visual-contract-replication.md`](cross-domain-visual-contract-replication.md)
- Claims boundary: [`docs/claims-audit.md`](../claims-audit.md)

## Executive finding

Stages 1-3 each hand-built a representation (a region, a resolution, an aggregation, a decoder) per property, per domain, and then measured how well it worked. This experiment asks a different question: given only a *declared evidence requirement* -- a component, a property, an evidence kind, a candidate region -- can a bounded, deterministic search **compile** an appropriate representation automatically, instead of a person picking one by hand?

Measured on 12 declared requirements across the two existing domains (arcade, warehouse), development=15 samples/category, held-out evaluation=40 samples/category:

- **11 of 12 requirements compiled.** The one exception, arcade alien target identity, correctly reports `insufficient_observability`: the alien sprite renders no identity marker at all, so no candidate in the bounded search -- or any search -- could recover it. This is the compiler distinguishing "the representation is missing" from "the evidence is missing," which is the entire point of having three outcomes instead of one pass/fail bit.
- **Both previously hand-fixed representation bugs were rediscovered automatically**, in both domains: the tank/robot max-aggregation tie (a max-aggregate candidate ties the true position with a 1px bleed column; the search ranks it below the mean/centroid candidates that don't tie, and rejects it outright for ambiguity) and the cooldown/door dilution bug (a naive whole-region decoder scores 0.000 at *every* resolution, including the finest; only a new development-only auto-narrowing decoder, fit without any privileged labels, recovers the signal).
- **Where the compiler and the historical manual representation both compile, they match exactly.** A genuine, live comparison -- evaluated on the identical held-out split as every other strategy, not cited from a different run -- shows the compiled and manual candidates achieving the same held-out accuracy in every one of the 11 compiling cases (1.000 in 10, 0.957 in 1). The compiler was never observed to do better or worse than hand engineering; it was observed to reach the same place without a person choosing the region/resolution/decoder combination by hand.
- **Naive reference strategies (fixed-coarse, always-pixel) are not a safe substitute for the search.** Both fail outright (0.000) on the two dilution cases, and both occasionally land on the max-aggregation candidate by the accident of an unordered tie-break, reproducing the ambiguity bug that a deliberate ranking avoids. This is not a design flaw in the reference strategies -- it is the intended demonstration that "just pick a resolution" is fragile in exactly the way the compiler's ranking (accuracy, then stability, then collision rate, then complexity) is not.

---

## Why this experiment was created

Stage 3 closed with an explicit, disciplined caveat: "the compiler selects resolution from requirements a person still writes by hand for each domain; it does not derive requirements from a renderer." This experiment does not remove that caveat -- a person still declares *what property, in what region, at what evidence kind* -- but it removes the next layer of hand-tuning: given that declaration, a person previously still had to pick the resolution, the aggregation, and the decoder, usually by trial and error (this is literally how the cooldown-dilution and tank-max-aggregation bugs were found and fixed in stages 1-2). The question here is:

> Given a declared evidence requirement, can a bounded, deterministic search select -- or construct, via a narrow, label-free repair step -- a representation that recovers the property, without a person iterating on resolution/aggregation/decoder by hand?

Three non-negotiable constraints governed the work throughout, verified before and after: (1) stages 1-3 remain untouched regression oracles (confirmed via `git status`, a bit-for-bit frame-hash tripwire, and the full stage 1-3 test suite passing unmodified at every checkpoint); (2) no privileged state reaches the compiler at decode time -- `decode_candidate` reads only rendered pixels; ground truth (`true_before`/`true_after`) exists solely for scoring and for the label-free `compute_dominant_fields` fit, which itself uses only pixel variance, never a label; (3) no architecture inflation -- the whole compiler is four small modules of dataclasses, a fixed candidate-generation dispatch table, and pure functions, with no new governance, lifecycle, or persistence layer, and no learned model.

---

## Repository location

```text
examples/visual_transition_benchmark/
    compiler/
        MANUAL_REPRESENTATION_INVENTORY.md   the 13-row inventory of every stage 1-3 hand-built representation
        contracts.py       VisualEvidenceRequirement -- the domain-neutral declared-evidence model
        candidates.py       RepresentationCandidate, RegionGeometry, bounded per-kind candidate generation
        evaluate.py         decode_candidate, compute_dominant_fields, evaluate_candidate
        compile.py          compile_requirement -- the deterministic selection policy, 3 outcomes
    compiler_adapters/
        arcade.py           6 declared cases against the existing arcade dataset
        warehouse.py        6 declared cases against the existing warehouse dataset
    compiler_run.py         CLI: compiles every case, measures held-out accuracy, 3 reference strategies
    tests/test_evidence_contracts.py, test_candidate_generation.py,
          test_representation_compilation.py, test_compiler_arcade.py,
          test_compiler_warehouse.py, test_compiler_identity.py,
          test_compiler_regressions.py, test_compiler_end_to_end.py
```

Nothing in `zeromodel/perception` changed. Nothing in `dataset.py`, `zeromodel_adapter.py`, `value_contracts.py`, `value_adapter.py`, `value_metrics.py`, `baselines.py`, `metrics.py`, `report.py`, `run.py`, `value_run.py`, the `domains/` package, `compilation/field_schema_compiler.py`, or their existing tests changed. `compiler/evaluate.py` reuses `compilation.field_schema_compiler.compile_field_schema` (P4A/P18A, unchanged) to build the actual field grid at each candidate's declared resolution; everything in `compiler/` is a decode/scoring/selection layer on top.

---

## The contract model

`VisualEvidenceRequirement` (`contracts.py`) declares, per property: `domain_name`, `component_type`, `property_name`, `evidence_kind` (one of `presence`, `numeric_value`, `categorical_state`, `spatial_position`, `signed_delta`, `exact_magnitude`, `relation`, `visible_identity`), a `candidate_region_id`, an optional `expected_value_domain`/`required_precision`, and a `comparison` rule. A `requirement_id` is a deterministic hash of every field, so identical declarations always produce the same identity. Validation is strict at construction: an unsupported `evidence_kind`/`comparison`, an empty identity field, a `numeric_value` with precision but no declared value domain, or a `visible_identity` without `permits_identity_marker=True` all raise immediately.

## Bounded candidate generation

`generate_candidates(requirement, region)` dispatches on `evidence_kind` to one of eight small generator functions, each varying only the parameters meaningful for that kind -- never an unbounded combinatorial product:

- `presence`: 2 candidates (cell-resolution, pixel-resolution) x 1 decoder (`presence_threshold`)
- `numeric_value` / `categorical_state`: 2 resolutions x 3 decoders (6 candidates) -- including `dominant_field_value`, the auto-narrowing repair
- `spatial_position`: 2 resolutions x 3 aggregations (6 candidates), 1 decoder (`argmax_field`)
- `signed_delta` / `exact_magnitude`: 2 resolutions x 2 aggregations (4 candidates)
- `relation`: 2 adjacency-threshold variants
- `visible_identity`: 2 candidates (a cell-mean fallback, and a pixel-level exact marker-pattern decoder)

Every case in this experiment considered between 2 and 6 candidates -- a person could enumerate this same list by hand in under a minute; the value is not search breadth, it is that the search runs the same way every time and never skips a candidate a person might have overlooked.

## The auto-narrowing repair (`dominant_field_value`)

The cooldown/door dilution bugs are not fixed by "use a finer resolution" alone: at *any* resolution, a decoder that averages the whole declared region dilutes 2 real signal pixels with 2 (or more) always-zero background pixels identically, because the region itself, not just the tile size, is wider than the true signal. `dominant_field_value` is a **development-only, label-free** fit: `compute_dominant_fields` walks only the development samples' pixel values (never `true_before`/`true_after`) and keeps whichever fields, within the declared region, ever carry non-background signal above a small epsilon; the decoder then averages only those. This is the one place the compiler does more than *select* among fixed candidates -- it constructs a narrower effective region from unlabeled pixel variance, which is exactly the mechanism a person used by hand when they built `value_contracts.py`'s narrow `VALUE_FIELD_SCHEMA` and `contracts.py`'s dedicated door-bar sub-region decoder.

## Selection policy and the three outcomes

`compile_requirement` evaluates every candidate on development samples only, then:

1. rejects any candidate below `min_decoding_accuracy` (0.95 throughout this experiment) or with unresolved collisions where exactness is required;
2. ranks the survivors by `(-decoding_accuracy, stability_false_change_rate, collision_rate, complexity_cost, candidate_id)` -- accuracy first, then stability under changes unrelated to the property, then ambiguity, then cost, with `candidate_id` as a final deterministic tie-break;
3. if any candidate passes, returns `compiled` with the top-ranked one;
4. otherwise, if *every* candidate's decoded output is degenerate (constant regardless of input, while the true value genuinely varies), returns `insufficient_observability`;
5. otherwise returns `insufficient_representation`, reporting the closest-scoring candidate for diagnostics.

Selection never looks at the held-out evaluation split.

---

## A gap found and fixed after the first merge (`a39c64d`)

The first version of this compiler (merged as `29d314e`) had two evidence gaps, both caught by external review after merge and fixed in a follow-up commit before this frozen run:

1. **The alien-identity case classified as `insufficient_representation`, not `insufficient_observability`.** `_nearest_level`, when no canonical levels were declared for a property (true of both identity cases, since neither has a fixed vocabulary of "levels"), fell back to returning the raw continuous intensity, `round(value, 6)`. That value genuinely varies across samples -- driven by scene content entirely unrelated to identity -- so it was never classified as *degenerate*, even though it carries zero identity information. The fix: with no declared canonical levels, `_nearest_level` now returns a fixed sentinel (`"no_canonical_levels_declared"`), which correctly collapses to a single constant value and lets the compiler recognize every candidate as degenerate. This changes nothing for any `numeric_value`/`categorical_state` case, all of which always declare real canonical levels in both adapters; it only affects the `visible_identity` cell-mean fallback candidate, and only when no marker-based candidate passes either. The warehouse crate-identity case is unaffected (it compiles via `local_marker_pattern` regardless of this candidate's classification).

2. **No comparison against the historical hand-built representation.** `compiler_run.py` originally compared only two naive, non-searched reference strategies (`fixed_coarse`, `always_pixel`). Review correctly noted this narrowed the supportable claim: the experiment could show the compiler finds *qualifying* candidates, but not that it matches or replaces manual engineering. `compiler_run.py` now also evaluates a `manual` strategy per case -- the literal historical resolution/aggregation/decoder from `MANUAL_REPRESENTATION_INVENTORY.md`, including the two cases (cooldown, door) where the manual fix hand-narrowed the *declared region itself* rather than auto-narrowing at decode time -- on the exact same held-out split as every other strategy. `alien_target_identity` has no manual entry: the inventory records no representation was ever successfully hand-built for it (a hidden/unobservable limitation, not a resolution a person picked), so there is nothing honest to compare against.

Both fixes are covered by tests (`test_compiler_arcade.py::test_alien_target_identity_reports_insufficient_observability`, `test_compiler_identity.py`, `test_compiler_end_to_end.py`'s manual-strategy assertions) and reflected in the numbers below.

---

## Environment and evaluated revision

- Commit: `aea9cab` (content-identical to `a39c64d`, the repair commit)
- Python 3.11.4, NumPy 2.2.3
- Command:
  ```
  python -m visual_transition_benchmark.compiler_run --dev-samples 15 --eval-samples 40 \
      --output-dir artifacts/evidence_contract_compiler
  ```
- Runtime: 841.0 seconds
- 12 declared requirements (6 arcade, 6 warehouse), development=15 samples/category, held-out evaluation=40 samples/category, disjoint seed ranges
- Tests: 125 tests in `examples/visual_transition_benchmark/tests/` (all stage 1-3 tests unmodified and passing, plus this stage's 8 new test files); 212 `packages/perception` tests unmodified and passing

### Per-case outcomes

| Domain | Case | Status | Selected | Held-out acc. | Fixed-coarse | Always-pixel | Manual |
|---|---|---|---|---:|---:|---:|---:|
| arcade | tank_presence | compiled | 3x4 `presence_threshold` | 1.000 | 1.000 | 1.000 | 1.000 |
| arcade | tank_position | compiled | 3x4 `argmax_field` (centroid) | 1.000 | 1.000 | 1.000 | 1.000 |
| arcade | tank_direction | compiled | 3x4 `signed_delta_over_position` | 1.000 | 1.000 | 1.000 | 1.000 |
| arcade | tank_movement_magnitude | compiled | 3x4 `exact_delta_over_position` | 1.000 | 1.000 | 0.853 | 1.000 |
| arcade | cooldown_value | compiled | 1x1 `dominant_field_value` | 1.000 | 0.000 | 0.000 | 1.000 |
| arcade | alien_target_identity | **insufficient_observability** | -- | -- | 0.000 | 0.000 | n/a |
| warehouse | robot_position | compiled | 6x6 `argmax_field` (centroid) | 1.000 | 1.000 | 0.000 | 1.000 |
| warehouse | robot_direction | compiled | 6x6 `signed_delta_over_position` | 1.000 | 1.000 | 1.000 | 1.000 |
| warehouse | robot_movement_magnitude | compiled | 6x6 `exact_delta_over_position` | 1.000 | 0.810 | 1.000 | 1.000 |
| warehouse | battery_value | compiled | 4x10 `nearest_permitted_value` | 1.000 | 1.000 | 1.000 | 1.000 |
| warehouse | door_state | compiled | 1x1 `dominant_field_value` | 0.957 | 0.000 | 0.000 | 0.957 |
| warehouse | crate_identity | compiled | 1x1 `local_marker_pattern` | 1.000 | 0.000 | 1.000 | 1.000 |

**Compiled = Manual in every one of the 11 compiling cases.** Full per-candidate development-split scores (every candidate considered, not just the selected one) are in `docs/results/evidence-contract-compiler-v1/compiler-results.json`.

### Why fixed-coarse and always-pixel sometimes fail, explained

- **Cooldown/door (0.000 at both naive resolutions):** the direct rediscovery target. A naive decoder averages the whole declared region regardless of tile size; the region is wider than the real signal (2 always-zero background columns alongside the cooldown/door bar), so dilution is identical at 4px and at 1px. Only `dominant_field_value`'s label-free auto-narrowing recovers the signal.
- **Tank movement magnitude / robot position (0.853 / 0.000 at "always_pixel"):** the naive reference strategy picks a decoder by name preference only, not by aggregation, and in these two cases it lands on the **max**-aggregation candidate -- reproducing the tank/robot max-aggregation tie by the accident of an unordered choice. This is the intended lesson, not a flaw in the comparison: "pick a resolution and a decoder kind" is not sufficient without also reasoning about aggregation, which is exactly what the compiler's ranking (by accuracy and stability, not by a fixed preference list) does automatically.
- **Robot movement magnitude / crate identity (0.810 / 0.000 at "fixed_coarse"):** the region's own coarse cell size either lands on a max-tie (robot) or cannot count discrete dots via a mean aggregate at all (crate identity requires the pixel-level `local_marker_pattern` decoder regardless of tile size).

---

## Threats to validity

- **Bounded candidate generation is bounded by design choice, not proof of completeness.** The search never considers a representation outside its 8 per-kind generator functions; a property whose correct representation needs a decoder kind not implemented here (e.g. a genuinely different aggregation primitive) would report `insufficient_representation` even if a person could still find a fix by hand. This experiment did not encounter such a case, but the search space's boundedness is a design decision, not a completeness guarantee.
- **Requirements are still hand-declared.** As in stage 3: a person still writes the `VisualEvidenceRequirement` (component, property, evidence kind, candidate region) for each property; the compiler selects a representation from that declaration, it does not derive the declaration from a renderer.
- **Twelve cases across two small, deterministic domains.** This is a within-domain search over hand-declared properties, not a claim that the same bounded candidate space would suffice for a substantially different rendering style (continuous positions, alpha-blended sprites, three-dimensional projection).
- **The `insufficient_representation` / `insufficient_observability` distinction has a real, now-documented edge.** It hinges on whether a decoder's output is classified *degenerate*; a decoder that produces a continuous, non-constant, but semantically meaningless value (as the pre-fix `nearest_permitted_value` fallback did) can spuriously avoid that classification. The fix here closes the one instance found; a different combination of evidence kind and decoder without a declared vocabulary could in principle reproduce the same edge case elsewhere.
- **The manual-baseline comparison is a live re-evaluation, not a re-derivation.** The literal manual candidate (resolution, aggregation, decoder, and for two cases, a hand-narrowed region) is reconstructed here from `MANUAL_REPRESENTATION_INVENTORY.md`'s description and evaluated fresh; it is not the original stage 1-3 code path re-run verbatim, though it is functionally identical to it.

---

## Supported claim

> Given a declared evidence requirement (component, property, evidence kind, candidate region), a bounded, deterministic, development-only search selected -- or, via a label-free auto-narrowing repair, constructed -- a representation that compiled for 11 of 12 declared requirements across two independently-built domains, including automatic rediscovery of two previously hand-fixed representation bugs. Where the compiled and literal historical manual representations both succeed, they achieve identical held-out accuracy in every case. The one non-compiling requirement (arcade alien target identity) was correctly classified as `insufficient_observability` -- evidence absent from the frame, not a representation the search failed to find.

Bounded to: the two tested domains, their 12 declared requirements, the frozen evaluation split (12 x 55 samples), the fixed candidate-generation scheme, and the recorded environment above.

## Claims to avoid

- "the compiler discovers evidence requirements" -- it compiles a representation from a requirement a person still declares; it does not infer what to look for from a renderer.
- "the compiler searches an unbounded space" -- generation is a fixed, small, per-kind dispatch (2-6 candidates per case here); it is not a general architecture search.
- "the compiler always distinguishes insufficient_representation from insufficient_observability correctly" -- one instance of exactly this misclassification was found and fixed in this stage; the underlying `is_degenerate` heuristic is not proven complete against every decoder/evidence-kind combination.
- "the compiler outperforms manual representation engineering" -- measured result is *parity*, not superiority: identical held-out accuracy in every one of the 11 compiling cases, never better.
- "this generalizes beyond small deterministic domains" -- two domains, twelve hand-declared properties; not evidence about continuous, learned, or naturalistic renderers.
