# Value-Aware Transition Contracts

- Implementation and reproduction: [`examples/visual_transition_benchmark/README.md`](../../examples/visual_transition_benchmark/README.md) (see "Stage 2")
- Frozen result record: [`docs/results/value-aware-transition-contracts-v1/`](../results/value-aware-transition-contracts-v1/)
- Prior stage: [`docs/research/visual-transition-debugging-benchmark.md`](visual-transition-debugging-benchmark.md)
- Claims boundary: [`docs/claims-audit.md`](../claims-audit.md)

## Executive finding

The prior benchmark ([visual-transition-debugging-benchmark.md](visual-transition-debugging-benchmark.md)) demonstrated that ZeroModel could attribute visible changes to declared components and detect selected stable/required-change contract violations, but explicitly identified **value-level transition reasoning** as "not demonstrated": whether a component changed in the correct direction, to the correct value, or against the correct target.

This experiment adds typed, decoded values and a small number of transition contracts on top of the *existing* P4A/P18A representation -- no new perception-package code, no new P-stage -- and measures whether that closes any of the three gaps.

Result: **two of the three gaps close, one does not.**

- **Wrong movement direction**: previously invisible (a wrong-direction move looks identical to a correct one at the component-label level). Now caught by an exact direction contract on **100%** of `tank_moves_wrong_direction` transitions.
- **Wrong numeric value**: previously inexpressible (presence/absence conformance has no notion of "value"). Now caught: wrong cooldown pixel-level values are flagged on **100%** of the two new cooldown-value fault categories; wrong movement magnitude (correct direction, wrong distance) is flagged on **100%** of `tank_moves_too_far` via an exact-magnitude relation.
- **Wrong target identity**: **still unresolved**. This environment renders only the current front-of-queue target, never the full alien list, so a legitimate hit and a hit that skips or misidentifies an alien produce pixel-identical evidence. No contract built only from frames + the action label can tell them apart.

The headline number, computed against the same evaluation split used to score all four systems:

> **1,178 of 1,800 faulty transitions were flagged "clean" by the component-level system (no missing/unexpected component finding) but are demonstrably value-wrong.**

This is the specific failure mode this experiment was built to expose: a correct component-level label can silently hide a wrong value.

---

## Why this experiment was created

The prior benchmark closed with three open capability gaps, stated explicitly in its own text:

```text
Did the tank move in the correct direction?
Did cooldown change to the correct value?
Did the expected target change?
```

Rather than proposing a new architecture stage to close them, this experiment asked the same kind of falsifiable question the prior one did:

> Can typed values and transition relations, decoded from the *existing* field representation, detect faults that the component-presence representation could not?

The constraint was explicit and self-imposed: keep the prior benchmark's dataset, tests, and runner completely unmodified; add only what is needed to test the new question; do not touch the perception package unless the benchmark proves a reusable, domain-independent abstraction is necessary. It did not turn out to be necessary -- everything here is `examples/visual_transition_benchmark/` code reusing `zeromodel.perception.fields`/`transition_evidence` at a finer grid resolution.

---

## Research question

> Can ZeroModel detect wrong movement direction, wrong cooldown value, wrong target removal, incorrect state delta, and a correct component change with an incorrect relation -- using only frame pairs and the action label, no hidden simulator state?

Four sub-questions, scored and reported **separately** from stage 1's component-attribution metric (per the explicit design requirement -- a component-level pass must never be allowed to imply a value-level pass):

- **Movement-direction accuracy** -- does decoded `sign(delta_x)` match the true simulated sign?
- **State-delta accuracy** -- does decoded `delta_x` match the true simulated delta exactly?
- **Cooldown-value accuracy** -- does the decoded cooldown level (`ready`/`blocked`/`out_of_domain`) match the true post-state?
- **Target-selection accuracy** -- does the decoded next-target column match the true post-state target? (Reported as a ground-truth comparison only; System D does not assert this as a pass/fail claim of its own, for reasons given below.)

---

## Repository location

```text
examples/visual_transition_benchmark/
    dataset.py            (additive changes only: VALUE_FAULT_CATEGORIES, build_value_transition, generate_value_episode/split)
    value_contracts.py     (new: pixel-decoded typed values + contracts)
    value_adapter.py       (new: System D)
    value_metrics.py       (new: stage-2 metrics)
    value_run.py           (new: stage-2 CLI)
    tests/test_value_*.py  (new: 24 tests)
```

Nothing in `zeromodel/perception` changed. Nothing in stage 1's `dataset.py` categories, `run.py`, `baselines.py`, `zeromodel_adapter.py`, `metrics.py`, `report.py`, or its own tests changed; all 24 stage-1 tests and all 212 `packages/perception` tests were re-run and still pass unmodified.

---

## Environment and evaluated revision

- Repository commit at evaluation time: working tree identical to commit `0e295a6` ("Can ZeroModel detect:"), parent `f28d8a1`
- Python 3.11.4, NumPy 2.2.3
- Command:
  ```
  python -m visual_transition_benchmark.value_run --dev-episodes 40 --eval-episodes 120 \
      --output-dir artifacts/value_aware_transition_contracts
  ```
- Runtime: 54.1 seconds
- Evaluation transitions: 2,760 total -- 2,160 reused from stage 1's frozen category set, 600 from the 5 new value-fault categories (120 per category, matching stage 1's per-category evaluation depth)
- Development transitions (40 episodes, both original and value-fault splits): generated only to verify episode-disjointness from evaluation; not used to select or calibrate any threshold in this experiment (every threshold, e.g. the cooldown tolerance and the alive/noise floor, is a fixed constant derived from the renderer's own documented pixel values, not fit to data)
- Warnings: 0
- Tests: 24 new (test_value_contracts.py, test_value_adapter.py, test_value_metrics.py, test_value_dataset.py, test_value_smoke_end_to_end.py) + 24 unchanged stage-1 tests = 48 passing; 212 `packages/perception` tests passing, unmodified

---

## The representation had to change resolution, not architecture

Two real bugs surfaced while building the value decoder, both against System C's *existing*, already-tested field schema (4px-wide tiles). Both are now regression-tested.

### Bug 1: tile dilution destroys an exact-value read

The cooldown indicator is a 2px-wide mark inside a 4px-wide field tile; the other 2 columns of that tile are always-zero background. Stage 1 only ever compared a tile's *before/after delta* (dilution cancels out of a delta -- a factor of 2 dilution on both sides of a subtraction is still proportional to the true change). Stage 2 needed the tile's **absolute** intensity (to classify it as "ready" vs. "blocked" vs. "out of contract"), and dilution corrupts an absolute read: the true 40/160 pixel levels decoded as 20/80, which don't match either canonical constant.

Fix: build a second, finer P4A field schema (1x1px tiles, `VALUE_FIELD_SCHEMA`) purely for value decoding. Same `build_transition_evidence_vpm` call, same package, just a different tile size for a different question.

### Bug 2: max aggregation hides which column is really lit

The tank sprite's rendered base is 5px wide -- 1px wider than its own 4px column cell -- and bleeds a single pixel into the neighboring cell. Aggregating a column's intensity by **max** across its rows reads that 1-pixel bleed as "fully lit," tying the bleed column with the true center column and making the decoded position ambiguous (observed as a wrong `delta_x` of -3 instead of -2 during development).

Fix: aggregate by **mean** across the band's rows instead of max. The true center column (3 of 3 rows lit) then reads unambiguously higher than the bleed column (1 of 3 rows lit).

### What this establishes about VPM field design

Neither bug was a defect in stage 1 -- stage 1's 4px-tile, max-aggregated schema is fully adequate for the question it answers ("did something change here?"). It became inadequate only when asked a different question ("what is the value here?"). The general point, not specific to this benchmark:

```text
presence question   -> coarse fields, delta comparison, may be sufficient
value question       -> fields sized to the smallest distinguishing feature, absolute-level comparison
direction question   -> a typed before/after relation over a decoded scalar
target-identity question -> requires observable identity-bearing evidence; not resolvable by finer fields alone
```

A VPM field schema is not a fixed, universal compression of an image -- it has to be compiled against the question being asked of it.

---

## Dataset design

### Reused, unmodified from stage 1

All 8 ordinary and 10 fault categories from the prior benchmark (`ORDINARY_CATEGORIES`, `FAULT_CATEGORIES` in `dataset.py`) are regenerated identically and re-scored under System D, to test whether value-awareness changes anything about categories stage 1 already characterized.

### New value-fault categories

Five new categories (`VALUE_FAULT_CATEGORIES`), each constructed by taking a real `TinyArcadeShooter.step()` transition and substituting a rendered value that is individually still "component-correct" but numerically wrong:

- `tank_moves_too_far` -- correct direction, tank rendered 2 cells away instead of 1
- `cooldown_activates_with_wrong_value` -- FIRE correctly triggers a change, but the cooldown pixel is set to an intensity (100) that is neither the ready (40) nor blocked (160) constant
- `cooldown_decreases_to_wrong_value` -- same corruption on the natural decay path
- `wrong_alien_disappears` -- a legitimate hit occurs, but the rendered next target is a column outside the true remaining alien queue entirely
- `two_aliens_disappear_instead_of_one` -- a legitimate hit occurs, but the render skips an extra alien, landing on a *valid-looking* but wrong next target

Two of the four requested fault categories are not new constructions, because this environment only ever renders one target at a time: "correct alien remains alive" is the same observable transition as stage 1's `fire_no_projectile`, and "target changes but not to expected state" is the same observable transition as stage 1's `alien_disappears_without_hit`. Both are reused (not re-implemented) and re-scored under the new value metrics in this experiment's evaluation set.

Every one of the 5 new categories was verified, across 200+ seeds, to be **component-label-correct** (`observed_changed_components == expected_changed_components`) -- i.e., stage 1's own metric would report these transitions as entirely unremarkable. That property is asserted by `tests/test_value_dataset.py::test_new_faults_look_correct_at_component_label_level`.

---

## Systems compared

Stage 1's three systems are reused unmodified. One system is added.

| System | Capability |
|---|---|
| A. Raw pixel difference | Detects visible pixel changes (unchanged from stage 1) |
| B. Privileged state baseline | Full ground truth (unchanged from stage 1) |
| C. ZeroModel (component-level) | Detects component presence/absence changes (unchanged from stage 1) |
| D. ZeroModel (value-aware) | Decodes typed values from pixels and evaluates direction/magnitude/cooldown-value/relation contracts (new) |

System D (`ValueAwareZeroModelAnalyzer` in `value_adapter.py`) wraps System C unchanged and adds a second, independent layer; its result type (`ValueTransitionAnalysis`) keeps `component_analysis` (System C's full, untouched result) and the new `values`/`verdict`/`value_flags` side by side, deliberately never merged into one score.

### Non-privileged input contract (unchanged discipline from stage 1)

`ValueAwareZeroModelAnalyzer.analyze()` has the identical signature to System C: `frame_before`, `frame_after`, `action`, `metadata` (only `transition_id`/`step_number`, no state). Every decoded value is read from pixels; `tank_x`/`target_x`/`cooldown` (the real simulator state) are used only as scoring ground truth in `value_metrics.py`, never passed to System D. This is verified by `tests/test_value_adapter.py::test_analyzer_signature_matches_component_analyzer_non_privileged_contract`.

---

## Value contracts

All four are evaluated from decoded pixel values and the action label alone:

- **Tank direction**: `sign(decoded delta_x) == sign(expected delta_x)`, where expected sign is `-1`/`+1`/`0` for `LEFT`/`RIGHT`/`STAY`|`FIRE` -- a pure action-conditioned claim, same discipline as stage 1's expectations.
- **Tank magnitude**: `decoded delta_x == expected delta_x` exactly (this environment's movement quantum is always exactly one cell). Strictly stronger than the direction check; catches faults that get the direction right but not the distance.
- **Cooldown value**: this environment's cooldown is a single binary flag, so its post-state is *fully* determined by the action alone regardless of its prior value -- `FIRE` always ends "blocked," anything else always ends "ready." This is a stronger, still non-privileged claim than stage 1 could make (stage 1 could only assert "cooldown changes," not what it changes *to*, because it never decoded the prior value).
- **Relation** (`alien_substitution_without_cooldown_blocked`): a legitimate alien substitution can only happen alongside a `FIRE` that ends cooldown "blocked" -- a genuine cross-field invariant of this environment's rules, checkable without any hidden state. It does not require declaring the *correct* target, only that a change of target must coincide with a blocked cooldown.

Target identity has **no contract** in System D by design: no combination of frame pixels and the action label can determine the correct next alien without the hidden queue, so System D does not assert one. Target-selection accuracy is reported only as a ground-truth comparison metric, never as a claim.

---

## Metrics

Reported in two families, kept explicitly separate (`value-benchmark-summary.md` never merges them into one score):

### Value-level accuracy (decoded value vs. true simulated state)

- Movement-direction accuracy
- State-delta accuracy (exact)
- Cooldown-value accuracy
- Target-selection accuracy (ground-truth comparison only, not a System D claim)

### Value-level fault localization (System D's own non-privileged flags vs. ground truth)

- Detection rate over transitions where *any* decoded dimension diverges from true state
- False-alarm rate over transitions with no such divergence
- `label_correct_but_value_wrong` -- the count of faulty transitions that are component-label-clean yet value-wrong; this is the number that justifies the experiment

---

## Aggregate results (2,760-transition evaluation split)

| Split | n | Direction acc. | Delta acc. (exact) | Cooldown acc. | Target acc. |
|---|---:|---:|---:|---:|---:|
| all | 2,760 | 0.870 | 0.826 | 0.826 | 0.783 |
| reused stage-1 categories | 2,160 | 0.833 | 0.833 | 0.889 | 0.833 |
| new value-fault categories | 600 | 1.000 | 0.800 | 0.600 | 0.600 |
| ordinary (non-faulty) | 960 | 1.000 | 1.000 | 1.000 | 1.000 |

Value-level fault localization (System D's own flags):

- **all**: detection rate 0.769 (n=1,560 value-faulty transitions), false-alarm rate 0.100 on 1,200 value-correct transitions
- **reused stage-1 categories**: detection rate 0.875 (n=960), false-alarm rate 0.100
- **new value-fault categories**: detection rate 0.600 (n=600), false-alarm rate n/a (0 value-correct transitions in this subset by construction)

Component-level metric (unchanged mechanism, shown for comparison, never conflated with the above): visible changed-component attribution micro-F1 -- pixel diff 0.956, privileged 1.000, ZeroModel (component-level) 1.000.

Hidden-failure headline: **1,178 of 1,800 faulty transitions** are component-label-clean (System C reports no missing/unexpected finding) yet value-wrong.

---

## Where value-awareness helped

### Wrong-direction movement (the primary target of this experiment)

`tank_moves_wrong_direction`: direction accuracy 0.000, delta accuracy 0.000 -- the tank visibly moves, in the wrong direction, and stage 1's component-level system reports this transition as entirely clean (`predicted=['tank']`, `missing=[]`, `unexpected=[]`). System D's direction contract flags `tank_direction_violation` on 100% of these transitions. See `stage1-eval-0000-0012.png` in the frozen result record.

### Wrong cooldown value

`cooldown_activates_with_wrong_value` / `cooldown_decreases_to_wrong_value`: cooldown accuracy 0.000 on both -- the cooldown region visibly changes, System C reports `predicted=['cooldown']` with no violation, but the pixel intensity (100) matches neither canonical level. System D's cooldown-value contract flags `cooldown_value_violation` on 100% of both categories. See `value-eval-0000-0001.png`.

### Correct label, wrong relation (magnitude)

`tank_moves_too_far`: direction accuracy 1.000 (the tank did move the right way) but delta accuracy 0.000 (it moved 2 cells, not 1). This is the concrete instance of "a component changes correctly but the relation is violated": the tank-magnitude relation flags `tank_magnitude_exceeds_single_step_bound` on 100% of these transitions even though the direction check alone would have passed it.

### A bonus the experiment did not originally target

Two of stage 1's own fault categories -- `alien_disappears_without_hit` and `unrelated_alien_change` -- are now also caught, at 100%, via the `alien_substitution_without_cooldown_blocked` relation: both render an alien-position change while the true state shows no cooldown activation, which the relation correctly flags as `relation:alien_substitution_without_cooldown_blocked`. Stage 1 could only place these in its soft "unexplained_change" bucket (attention-worthy, not a violation); stage 2 promotes them to a hard, correctly-reasoned flag.

---

## Where value-awareness remains blind

### Target identity

`wrong_alien_disappears`, `two_aliens_disappear_instead_of_one`, and stage 1's `fire_no_projectile`: target-selection accuracy 0.000 on all three, value-fault detection rate 0.000 on the first two. Every one of these renders a cooldown-blocked, single-target substitution that is pixel-identical to a legitimate hit. The `alien_substitution_without_cooldown_blocked` relation is satisfied (cooldown genuinely is blocked) in every one of these faults, so it correctly does not fire -- there is nothing wrong with the *relation*, only with which specific alien was chosen, and that information does not exist anywhere in the two frames or the action label.

This is not a shortcoming of the contract design; it is the environment's visual bandwidth. A single-target renderer cannot distinguish "correct alien removed," "wrong alien removed," and "two aliens removed" from each other without either (a) rendering more state than it currently does, or (b) accepting a privileged input the deployable system is not allowed to have.

---

## Four capability levels (extends stage 1's three-level framework)

Stage 1 identified three levels and showed only the first was demonstrated. This experiment adds a fourth axis and shows two of the remaining three are now demonstrated, one is not:

### Presence-level transition reasoning (stage 1, unchanged)
`Did this declared component change?` -- demonstrated, unchanged.

### Value-level transition reasoning (previously not demonstrated; now partially demonstrated)
```text
Did the tank move in the correct direction?     -> demonstrated
Did cooldown change to the correct value?        -> demonstrated
Did a field change by the required exact amount? -> demonstrated (magnitude relation)
```

### Relational transition reasoning (new in this experiment; demonstrated for one cross-field invariant)
`Does a change in one component correctly coincide with the required state of another?` -- demonstrated for the alien/cooldown invariant; not attempted for other combinations.

### Target-identity reasoning (previously not demonstrated; still not demonstrated)
`Did the expected target change?` -- not demonstrated. Requires observable identity-bearing evidence this environment does not render.

---

## Visual artifacts

Representative diagnostic panels (5 curated examples; the full generated corpus of 610 PNGs is not committed) are in `docs/results/value-aware-transition-contracts-v1/representative-artifacts/`:

- `wrong-direction-detected.png` -- `tank_moves_wrong_direction`, System D catches what System C's label-only view could not
- `wrong-cooldown-value-detected.png` -- `cooldown_activates_with_wrong_value`
- `wrong-magnitude-detected.png` -- `tank_moves_too_far`
- `wrong-target-unresolved.png` -- `wrong_alien_disappears`, an honest miss
- `fire-no-projectile-unresolved.png` -- stage 1's original blind spot, confirmed still unresolved under value-awareness

---

## Test coverage

24 new tests across 5 files:

- `test_value_dataset.py` -- stage-1 categories untouched, new categories disjoint and deterministic, new faults verified label-correct, no degenerate zero-visible-change cases
- `test_value_contracts.py` -- regression tests for both representation bugs (tile dilution, max-aggregation tie), decode-vs-ground-truth agreement on ordinary transitions, contract catches for each new fault, honest blindness to target-identity faults, zero false alarms across 30+ seeds per non-boundary ordinary category
- `test_value_adapter.py` -- component and value layers independently correct on the same transition, deterministic repeat analysis, non-privileged signature parity with System C
- `test_value_metrics.py` -- accuracy and localization metric unit tests, including the `label_correct_but_value_wrong` hidden-failure counter
- `test_value_smoke_end_to_end.py` -- small combined dataset through both metric families end to end

All 24 stage-1 tests and all 212 `packages/perception` tests pass unmodified alongside these.

---

## Threats to validity

Carried forward from stage 1, plus two new to this experiment:

- **Controlled synthetic domain** -- as stage 1.
- **Hand-declared component/value mapping** -- the four decoded quantities (tank column, alien column/aliveness, cooldown level) are read against constants taken directly from the renderer's own drawing code, not detected or learned.
- **Boundary false alarms carried forward** -- `tank_remains_stationary_at_boundary` (a legitimate transition) is flagged by the direction/magnitude contracts for the same reason it was flagged by stage 1's presence contract: neither has a notion of a screen edge without privileged state. This inflates the false-alarm rate by a known, fixed, documented amount (10% of ordinary transitions in this environment).
- **Fixed, not tuned, thresholds** -- the cooldown tolerance (±15/255) and the alive/noise floor (0.05) are set from the renderer's own documented constants and were not calibrated against the development split; a different renderer would need its own constants re-derived the same way.
- **Single relation checked** -- only one cross-field relation (alien substitution requires cooldown-blocked) was implemented; this is not an exhaustive relational contract language.

---

## What this experiment demonstrates

- Adding typed, pixel-decoded values on top of the existing P4A/P18A representation -- at a finer field resolution, with mean instead of max aggregation, no new perception-package code -- resolves the wrong-direction blind spot stage 1 explicitly reported as open.
- The same extension adds two further capabilities stage 1 had no vocabulary for at all: exact-magnitude checking and absolute cooldown-value checking.
- A single, simple cross-field relation recovers hard-violation status for two fault categories stage 1 could only place in its soft "unexplained" bucket.
- 1,178 of 1,800 faulty transitions in this evaluation split are component-label-clean yet value-wrong, demonstrating concretely why component-level and value-level results must be reported separately.

## What this experiment suggests

- Presence/absence conformance and value correctness are complementary layers of the same representation, not competing designs -- a production system needs both.
- The correct field resolution and aggregation rule for a VPM depend on the question being asked of it; the same environment needed a coarser, delta-comparing schema for presence and a finer, absolute-value-comparing schema for value.

## What this experiment does not establish

- General visual perception, causal diagnosis, or semantic understanding from pixels (unchanged from stage 1).
- Target/alien identity correctness -- this remains unresolved and is not expected to resolve without either richer non-privileged observations (e.g. rendering the full alien queue) or accepting a different, explicitly-scoped privileged input.
- A general relational-contract language -- one relation was implemented and validated; this is not evidence that arbitrary cross-field relations are equally easy to specify or equally reliable.
- Generalization beyond the tested renderer, component schema, and fixed decoding constants.

---

## Architecture implications

This experiment required no new perception-package code and no new P-stage. It reused, at a finer resolution:

```text
fields (P4A) -- same build_grid_field_schema, tile_width=1 instead of 4
transition evidence (P18A) -- same build_transition_evidence_vpm
```

It did not require, and did not add:

```text
new PerceptionRegionAnnotationDTO semantics
transition_conformance.py (P18B) -- value contracts are evaluated directly against decoded values, not through P18B's expectation/finding machinery
transition_discovery.py (P18C)
any candidate promotion, materialization, activation, rollback, certification, admission, or governance stage
```

This reinforces stage 1's own architecture conclusion: the core visual testing instrument (fields, evidence, expectations, conformance, discovery) contains the demonstrated research value; the governed lifecycle layer (promotion through governance) remains unexercised and unneeded by either benchmark.

---

## Supported claim

The strongest claim supported by the frozen result is:

> Within the deterministic TinyArcadeShooter benchmark, typed visual field values and transition relations, decoded from the existing P4A/P18A representation at a finer field resolution, detected wrong-direction, wrong-cooldown-value, and wrong-magnitude faults that were invisible to the earlier component-change representation.

A bounded negative claim is equally supported and must be stated alongside it:

> The current non-privileged visual representation does not identify the correct hidden target when target identity is determined by an unobserved queue or state unavailable in the supplied frames.

Both claims must remain bounded to the deterministic renderer, the declared component/value schema, the frozen dataset, and the recorded system configurations.

---

## Claims to avoid

Do not claim:

- "ZeroModel understands game semantics" -- it decodes four specific, hand-declared scalar quantities against hand-declared constants.
- "value-aware ZeroModel resolves target identity" -- it explicitly does not, and the experiment measures exactly how often it fails to.
- "the representation bugs found here are fixed for all VPM field schemas" -- they were fixed for this environment's specific geometry (2px cooldown mark in a 4px tile, 5px tank base in a 4px cell); a different renderer needs its own resolution analysis.
- "value-aware transition contracts generalize beyond this benchmark" -- no other domain has been tested.
- "component-level and value-level results can be summarized as one score" -- the entire point of this experiment is that they must not be.
