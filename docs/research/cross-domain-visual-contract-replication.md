# Cross-Domain Visual Contract Replication

- Implementation and reproduction: [`examples/visual_transition_benchmark/`](../../examples/visual_transition_benchmark/) (`domains/`, `compilation/`, `cross_domain_*.py`)
- Frozen result record: [`docs/results/cross-domain-visual-contracts-v1/`](../results/cross-domain-visual-contracts-v1/)
- Prior stages: [`visual-transition-debugging-benchmark.md`](visual-transition-debugging-benchmark.md), [`value-aware-transition-contracts.md`](value-aware-transition-contracts.md)
- Claims boundary: [`docs/claims-audit.md`](../claims-audit.md)

## Executive finding

Both prior stages were measured on one renderer: `TinyArcadeShooter`. Every capability claim carried an implicit qualifier -- "in this fixture" -- that had never been tested. This experiment builds a second, unrelated environment (a small Sokoban-like warehouse: a robot, identity-marked crates, a door, a battery, walls) from scratch, runs the same declared systems and the same metric functions against it, and asks which capabilities survive contact with a domain that differs on purpose: discrete 2D movement instead of 1D, strong inter-object relations (pushing, adjacency, occupancy) instead of mostly-local rules, multiple persistent identity-bearing objects instead of one visible target.

Result, measured on 1,800 arcade and 2,300 warehouse evaluation transitions (100 per category in both):

- **Replicated in both domains**: visible component attribution (micro-F1 1.000/1.000) and the one implemented cross-field relation (1.000/0.957).
- **Domain-specific**: unexpected-change detection (0.500 arcade / 1.000 warehouse) and missing-change detection (0.692 / 1.000) -- both real, both meaningfully measured, but arcade's own declared expectations are more conservative than warehouse's, producing different hit rates on the same underlying mechanism.
- **Not replicated at the pre-declared 0.90 bar, but replicated as a *pattern***: direction and value correctness cluster at 82-91% in *both* domains, dragged down by the identical class of case in both -- a legitimate transition (a movement blocked by a wall, a push that cannot proceed) that a frame+action-only decoder cannot distinguish from a failed command. That the *shortfall* itself reproduces, at a similar rate, in an unrelated domain is the more informative result than either raw number.
- **Measured, not solved**: visible crate identity is decodable (25.0% correct against ground truth on the 800 transitions where it applies) but not resolvable by any contract built from frames + action alone -- confirmed structurally, not just empirically.

The extraction exercise (Stage A) also produced its own verdict: the shared runner (`cross_domain_metrics.py`, most of `metrics.py` reused unchanged) contains no arcade or warehouse component name, both domains implement the identical `VisualTransitionDomain` protocol, and the arcade wrapper is proven bit-for-bit identical to stage 1/2's own pipeline. One piece of stage-1 machinery did **not** transfer as-is: P18B's declarative conformance model assumes disjoint, static semantic regions, which arcade has and warehouse does not (robot and crates share every possible cell). This is recorded as an architecture finding, not a defect.

---

## Why this experiment was created

Stage 2 closed with a specific, disciplined recommendation: strengthen the value-aware direction narrowly, and do not generalize the result until it had been tested somewhere else. This experiment is that test, not a new feature:

> Does the visual-contract method survive when the renderer, objects, actions, state variables, and transition rules are different -- or does "ZeroModel improves on pixel differencing" only hold for one hand-built fixture?

The instruction governing this experiment was explicit and is treated as a hard constraint throughout: **the arcade benchmark must remain a regression oracle while the second domain is added.** Every stage-1/2 file is untouched; the arcade domain is a translation layer, not a rewrite.

---

## Repository location

```text
examples/visual_transition_benchmark/
    domains/
        protocol.py             domain-neutral seam (no arcade/warehouse names)
        arcade/domain.py        thin wrapper around stages 1-2, unmodified
        warehouse/
            model.py            new, from scratch: robot/crates/door/battery grid
            rendering.py        geometric-glyph renderer, no external assets
            dataset.py          privileged ground-truth masks + record type
            faults.py           23 categories: 8 ordinary + 15 faults across 4 families
            contracts.py        component- and value-level decoding/conformance
            domain.py           protocol implementation
    compilation/
        evidence_requirements.py
        field_schema_compiler.py
    cross_domain_baselines.py   domain-neutral System A / System B
    cross_domain_metrics.py     domain-neutral value-level capability scoring
    cross_domain_run.py         CLI: generates, analyzes, scores, reports both domains
    tests/test_cross_domain_*.py, test_warehouse_*.py, test_field_schema_compiler.py
```

Nothing in `zeromodel/perception` changed. Nothing in `dataset.py`, `zeromodel_adapter.py`, `value_contracts.py`, `value_adapter.py`, `value_metrics.py`, `baselines.py`, `metrics.py`, `report.py`, `run.py`, `value_run.py`, or their existing tests changed.

---

## Environment and evaluated revision

- Commit: `f7652d4` ("research(perception): add cross-domain visual contract replication"), parent `6b75b15`
- Python 3.11.4, NumPy 2.2.3
- Command:
  ```
  python -m visual_transition_benchmark.cross_domain_run \
      --arcade-dev-episodes 40 --arcade-eval-episodes 100 \
      --warehouse-dev-episodes 20 --warehouse-eval-episodes 100 \
      --output-dir artifacts/cross_domain_visual_contracts
  ```
- Runtime: 397.1 seconds
- Evaluation transitions: 1,800 arcade (18 categories x 100), 2,300 warehouse (23 categories x 100)
- Development episodes generated only to verify episode-disjointness from evaluation (40 arcade, 20 warehouse); never analyzed, never used to pick a threshold
- Tests: 100 total in `examples/visual_transition_benchmark/tests/` (all stage-1/2 tests unchanged and passing, plus this stage's new tests); 212 `packages/perception` tests unmodified and passing

### A real gap found and fixed during execution

The first full run reported `missing_change_detection_rate = 0.000` for warehouse with `n_relevant = 0` -- not a measured failure, an empty test set: none of the original 22 warehouse categories constructed a case where an expected region shows *zero* pixel difference (arcade's `fire_no_projectile` analogue). A 23rd category, `push_fails_silently` (a valid push renders as an exact no-op), was added, the full test suite re-run (80/80, then 100/100 after this stage's own tests), and the evaluation re-executed at the same scale. The number reported below (1.000 warehouse, `n_relevant = 100`) is from that corrected run. This is recorded here rather than silently fixed because it is exactly the kind of empty-denominator artifact the whole exercise is designed to catch.

---

## Stage A: the domain-neutral seam

`domains/protocol.py` defines `VisualTransitionDomain`, `DomainTransition`, `ComponentAnalysisResult`, `ValueAnalysisResult`. `DomainTransition`'s field names deliberately mirror `dataset.TransitionRecord` (`observed_changed_components`, `expected_changed_components`, `is_faulty`, ...), which means stage 1's own metric functions run **unmodified** against transitions from either domain:

- `metrics.component_multilabel_metrics`
- `metrics.unexpected_change_summary`
- `metrics.missing_change_summary`
- `metrics.false_implicated_components`
- `metrics.field_precision_recall`

This is not a re-implementation with the same name; it is the literal same function, imported once, called twice. Value-level ground truth and decoded values use one small shared vocabulary (`*_expected_sign`/`*_decoded_sign` for direction, `*_expected_delta`/`*_decoded_delta` for magnitude, `*_expected_level`/`*_decoded_level` for any named numeric/categorical channel, `relation_expected_satisfied`/`relation_decoded_satisfied`, `identity_expected_id`/`identity_decoded_id`) so `cross_domain_metrics.py` can score both domains without ever branching on a component name.

**Verified, not asserted**: `test_cross_domain_arcade_regression.py` regenerates transitions and re-runs both stage-1/2 analyzers directly, then through `ArcadeTransitionDomain`, and asserts every frame, category, fault flag, predicted-field set, and decoded value is identical.

---

## Stage B: the warehouse environment

Deliberately small, per instruction: a 5x5 walled grid with a 3x3 interior, one door (`(2,2)`), one goal (`(3,3)`), up to 3 crates with visible identity markers (1/2/3 fixed corner dots), a 3-segment battery strip. Rendered with plain geometric glyphs (filled squares, rings, bars) at 6px/cell, no external assets -- `domains/warehouse/rendering.py`.

Actions: `MOVE_UP/DOWN/LEFT/RIGHT`, `PUSH_UP/DOWN/LEFT/RIGHT`, `OPEN_DOOR`, `WAIT`. Adaptation note: the prompt's generic `PUSH` is implemented as four direction-specific actions, exactly mirroring `MOVE_*`, because an undeclared-direction push is not a deterministic action on a 2D grid.

23 categories: 8 ordinary, 4 presence faults, 4 value faults, 3 relation faults, 3 identity faults, plus `push_fails_silently` (added during execution, see above). Every fault category was verified, across many seeds, to either be component-label-correct (value/identity/push-based-relation faults) or genuinely presence-detectable (the rest) -- `test_warehouse_dataset.py`.

### Two real representation bugs found building the decoder (both regression-tested)

1. **Tile dilution, again, in a new shape.** The door's bar glyph is only 2px wide inside its 6px cell; whole-cell mode classification reads the cell's majority-empty background and misclassifies every door state as "empty" regardless of open/closed. Fixed with a dedicated sub-region decoder (`classify_door_state`), exactly the same class of fix stage 2 needed for the cooldown indicator.
2. **A canonical-level tolerance collision.** `wall` (60) and `goal_ring` (50) are only 10 apart; an initial 10/255 tolerance window matched both simultaneously at the exact boundary, and dict-iteration order silently picked the wrong one for every wall cell. Fixed by switching to nearest-match classification with a tolerance safely under half the minimum gap (4/255).

### A domain constraint discovered, not designed in

The renderer draws crates, then the robot, on top. A successful push always lands the robot exactly on the crate's *pre-push* cell. If a crate fails to follow (the `push_advances_robot_without_crate` fault), the robot visually occludes it -- unrecoverable from pixels alone, by any pixel-based system, for any resolution. This is documented in `faults.py` and confirmed in `test_warehouse_contracts.py` as an honest, structural blind spot, distinct from the decoder bugs above (those were fixed; this one cannot be, without changing what the renderer draws).

---

## Systems compared (both domains, same functions)

| System | Arcade | Warehouse |
|---|---|---|
| A. Raw pixel difference | `baselines.pixel_diff_baseline` (unchanged) | `cross_domain_baselines.pixel_diff_baseline` (same algorithm, domain-declared band masks) |
| B. Privileged state | `baselines.privileged_baseline` (unchanged) | `cross_domain_baselines.privileged_baseline` (same construction) |
| C. ZeroModel, component-level | `zeromodel_adapter.ArcadeBandZeroModelAnalyzer` (unchanged) | `contracts.WarehouseComponentAnalyzer` (new; direct Python conformance, see below) |
| D. ZeroModel, value-aware | `value_adapter.ValueAwareZeroModelAnalyzer` (unchanged) | `contracts.WarehouseValueAnalyzer` (new) |

For warehouse's naive pixel-diff baseline, the declared component bands for "robot" and "crate" are necessarily identical (the full interior -- a domain-unaware operator has no other static region to declare), which is itself part of the measurement: it shows what happens to a naive baseline when object regions genuinely overlap, not an artificially crippled comparison.

---

## Metrics

Presence-level: reused `metrics.py` functions, unchanged (see Stage A). Value-level: `cross_domain_metrics.py`'s five generic capability checks (direction, magnitude, value, relation, identity), each returning `None` ("not applicable to this transition") rather than a score when the domain has no ground truth for that channel -- exactly how arcade's `identity_expected_id` is always absent.

---

## Results

### Component attribution and presence-level conformance (100/category, both domains)

| Metric | Arcade | Warehouse |
|---|---:|---:|
| Component attribution micro-F1 -- pixel diff | 0.937 | 0.863 |
| Component attribution micro-F1 -- ZeroModel | 1.000 | 1.000 |
| Component attribution exact-set accuracy -- pixel diff | 0.889 | 0.522 |
| Component attribution exact-set accuracy -- ZeroModel | 1.000 | 1.000 |
| Unexpected-change detection rate (n_relevant) | 0.500 (600) | 1.000 (500) |
| Missing-change detection rate (n_relevant) | 0.692 (325) | 1.000 (100) |
| False-implicated components, mean | 0.181 | 0.217 |

The pixel-diff baseline's exact-set accuracy drop (0.889 -> 0.522) is the sharpest single number in this experiment: a domain-unaware pixel operator's component attribution degrades substantially once object regions genuinely overlap, while ZeroModel's occupant-classification step (a value-level decode used *underneath* presence-level judgment -- see Architecture implications) holds at 1.000 in both.

### Value-level capability rates (n_applicable in parentheses)

| Capability | Arcade | Warehouse |
|---|---:|---:|
| Direction correctness | 0.833 (1,800) | 0.870 (2,300) |
| Magnitude correctness | 0.833 (1,800) | 0.826 (2,300) |
| Value correctness (cooldown / battery+door) | 0.889 (1,800) | 0.826 (2,300) |
| Relation correctness | 1.000 (1,800) | 0.957 (2,300) |
| Identity correctness | unavailable | 0.250 (800) |
| Hidden value faults (label-clean, value-wrong) | 175 / 1,000 faulty | 900 / 1,500 faulty |

---

## Where replication held

**Component attribution and the one implemented relation.** Both are at or above 0.95 in both domains, using the identical scoring function, on an environment with fundamentally different object-overlap geometry. This is the strongest result: the P4A/P18A representation plus a domain-declared occupant classifier attributes visible change correctly regardless of whether components occupy disjoint or shared regions.

**The hidden-fault pattern.** Both domains show a large share of faulty transitions that are component-label-clean yet value-wrong (17.5% arcade, 60.0% warehouse -- the warehouse rate is higher because more of its fault families were deliberately built to be label-correct, per the fault-family design in Stage B). The *existence* of this gap, not its exact size, is what replicates.

## Where replication did not hold at the declared bar, and what that means

**Direction, magnitude, and value correctness all land in the low-to-high 80s in both domains**, short of the 0.90 bar set before running the evaluation. Inspection of the transition-level results attributes this consistently, in both domains, to the same cause: a legitimate transition where the commanded action structurally implies a change (movement, a push) but a boundary/precondition prevents it (`tank_remains_stationary_at_boundary` / `robot_blocked_by_wall`, `push_attempt_with_no_crate_is_noop`, `push_fails_silently`). Neither domain's value contract has a notion of "the edge of the world" or "nothing to push" without privileged state, so both structurally score these as violations. Because this shortfall reproduces at a similar magnitude in an unrelated domain, the correct read is not "the value layer is unreliable" but "frame+action-only value contracts have a structural false-alarm rate on precondition-blocked actions, and that rate is now measured in two domains instead of asserted from one."

**Identity remains unresolved, and is now understood to be a representation question, not an analysis question.** 25.0% correct (below the ~50% a 2-crate coin flip would produce, because several fault categories were built specifically to present a wrong marker) confirms that a value-aware decoder reading pixels correctly still cannot recover *which* persistent object moved without an observable channel dedicated to identity that a hidden queue does not corrupt. Section 15's stop condition ("the current method is consistently inferior... a negative result is valid") applies here directly: no further contract sophistication is warranted before the representation question is resolved.

---

## Architecture implications

### What replicated across the compiler

Given declared evidence requirements, `compile_field_schema` reproduces stage 1's exact coarse tile size (4x1px) and stage 2's exact fine tile size (1x1px) from requirements describing what those stages needed, and independently derives a 1x1px schema for warehouse from a different requirement (battery segments) -- `test_field_schema_compiler.py`. The compiler genuinely selects representation from declared requirements; it is not a wrapper that happens to agree with hindsight.

### What P4A/P18A carried over unchanged

`build_grid_field_schema` and `build_transition_evidence_vpm` are the same two calls in both domains, at whatever tile size the compiler selects for that domain's declared requirements.

### What did not carry over: P18B

P18B's `evaluate_transition_conformance` assumes annotations are pre-known, disjoint semantic regions -- true in arcade (tank/alien/cooldown/background never share a pixel), false in warehouse (robot and any crate can occupy the same cell). Presence-level judgment in the warehouse domain is therefore direct Python over an occupant classification derived from decoded pixel intensity -- a value-level step underneath a presence-level judgment, which is itself the clearest instance of this experiment's central architectural point:

```text
presence question            -> coarse fields may be sufficient (arcade)
                                 or may first require an occupant-level value decode (warehouse)
value question                -> fields sized to the smallest distinguishing feature (cooldown, door)
direction/relation question    -> a typed before/after comparison over decoded scalars
identity question              -> requires an observable, corruption-resistant identity channel;
                                   not resolvable by finer fields or better aggregation alone
```

### Domain-specific adapters that were unavoidable

- Warehouse's occupant classifier (mode-based, per grid cell) and its dedicated door/battery decoders -- direct Python, not P18B, for the reason above.
- Warehouse's cross-field relation (`crate_change_without_robot_adjacency`) -- a genuine new predicate, not a reuse of arcade's `alien_substitution_without_cooldown_blocked`, though both follow the same declared, non-privileged, action-and-decoded-value-only shape.

---

## Test coverage

100 tests total in `examples/visual_transition_benchmark/tests/`, including (new to this stage):

- `test_cross_domain_arcade_regression.py` -- proves the arcade wrapper changes nothing observable
- `test_field_schema_compiler.py` -- proves the compiler reproduces stage 1/2's hand-built schemas from declared requirements, and rejects a requirement that resolves to zero fields
- `test_warehouse_dataset.py`, `test_warehouse_contracts.py` -- determinism, label-correctness-by-family, the two representation-bug regressions, no false alarms outside the two documented structural cases
- `test_cross_domain_metrics.py` -- the five generic capability checks, including "not applicable" handling
- `test_cross_domain_smoke_end_to_end.py` -- the full pipeline on a small deterministic sample

All pre-existing stage-1/2 tests (48) and `packages/perception` tests (212) pass unmodified.

---

## Threats to validity

- **Two domains is a replication, not a generalization proof.** A third, differently-shaped domain (continuous positions, more than two persistent identity-bearing objects, asynchronous events) could still fail differently.
- **Hand-declared evidence requirements.** The compiler selects resolution from requirements a person still writes by hand for each domain; it does not derive requirements from the renderer automatically.
- **The z-order occlusion finding is renderer-specific.** A different draw order (robot under crates, or alpha-blended) would change which faults are visually recoverable; it does not change the general finding that *some* renderer choice will always create at least one occlusion class.
- **Threshold choice (0.90).** The specific bar used to declare "replicated" is a judgment call recorded before running the evaluation; the raw rates are reported alongside it precisely so a reader can apply a different bar.
- **Identity accuracy denominator.** The 800-transition denominator mixes ordinary transitions (where identity decoding is easy) with fault categories specifically built to present a wrong marker; the 25.0% figure should not be read as "identity decoding usually fails at pixels," only as "identity *correctness against ground truth* is low across this mixed evaluation set," which is the intended, and sufficient, measurement for this experiment's purpose.

---

## Supported claim

> Within two independently-implemented deterministic visual domains sharing no rendering code, component-level visual contracts (presence attribution, one cross-field relation) replicated at or above a 0.95 bar using the identical scoring functions. Value-level contracts (direction, magnitude, numeric/categorical value) did not clear a pre-declared 0.90 bar in either domain, but their shortfall reproduced at a similar magnitude and from the same structural cause in both, which is itself evidence of a general, frame-plus-action-only limitation rather than a fixture-specific defect. Visible object identity was measurably decodable but not resolvable from either domain's frames and action label alone.

Bounded to: the two tested renderers, their declared component/value schemas, the frozen datasets (1,800 + 2,300 transitions), and the recorded system configurations.

## Claims to avoid

- "ZeroModel generalizes across visual domains" -- two domains were tested, both small and deterministic; this is a replication of specific capability classes, not a generality proof.
- "value-level contracts are unreliable" -- the measured shortfall is specific to precondition-blocked actions and reproduces consistently; it is not evidence that value contracts fail broadly.
- "identity resolution requires more sophisticated analysis" -- the finding is the opposite: analysis sophistication is not the bottleneck; the representation needs an identity-bearing observable channel that survives the fault, which neither domain currently provides in a corruption-resistant way.
- "the compiler discovers the correct representation automatically" -- it compiles *declared* requirements into a schema; it does not infer requirements from a renderer.
- "P18B is obsolete" -- it remains exactly correct for domains with disjoint static semantic regions (arcade); it was not designed for, and does not claim to handle, overlapping dynamic regions.
