# Visual Transition Debugging Benchmark

Answers one question: **can ZeroModel identify which visual components changed,
or were relevant to a state transition, more accurately or more usefully than
straightforward visual baselines?** It is a benchmark, not a new perception
stage. No P18H/P19, no governance/certification/promotion/lifecycle machinery
was added.

Run it:

```bash
PYTHONPATH=examples python -m visual_transition_benchmark.run \
    --dev-episodes 40 --eval-episodes 120 \
    --output-dir artifacts/visual_transition_benchmark
```

Run the tests:

```bash
PYTHONPATH=examples python -m pytest examples/visual_transition_benchmark/tests -q
```

## Implementation note (what is reused, what is excluded)

**Domain** (reused unmodified): `zeromodel.video.arcade_policy` -- `model.py`'s
`TinyArcadeShooter` (the real game-rule engine; every "true" transition is
produced by actually calling `.step()`, never reimplemented) and
`rendering.py`'s `render_state_frame` (the real renderer; every frame is
produced by actually calling it, never hand-drawn).

**Perception** (reused unmodified, from `zeromodel.perception`):

| Stage | Module | Used for |
|---|---|---|
| P1  | `representation.py` | Canonical PNG encoding of raw frames |
| P4A | `fields.py` | Deterministic 4x1px field partition of the 16x28 canvas |
| P6  | `expectations.py` | `PerceptionRegionAnnotationDTO` for the 4 declared static bands |
| P18A | `transition_evidence.py` | Per-field before/after change measurement |
| P18B | `transition_conformance.py` | Action-conditioned expectation checking (confirmed / missing / unexpected / unexplained) |
| P18C | `transition_discovery.py` | Secondary demo: recurrence of unexplained fields across one episode's cohort |

**Excluded**: every certification/governance/promotion/lifecycle stage
(P12-P17, P18D-P18G). None of it answers this benchmark's question; adding it
would just be re-adding the machinery the task explicitly asked not to build.

**Privileged ground truth is isolated from the deployable adapter**: exact
per-transition component masks (`dataset.py`'s `tank_mask`/`alien_mask`/
`cooldown_mask`, built from `tank_x`/`target_x`, i.e. real game state) are used
only by the dataset generator (to label ground truth) and System B (the
privileged baseline). `zeromodel_adapter.py` (System C) never imports or sees
`tank_x`/`target_x`/`cooldown` -- it only ever sees `frame_before`,
`frame_after`, `action`, and a `TransitionMetadata` containing just
`transition_id`/`step_number`. Its 4 named regions ("tank", "alien",
"cooldown", "background") are static row/column bands read once from
`rendering.py`'s own drawing code, identical for every transition -- see the
adapter module docstring for the exact bands and why this is not a
ground-truth leak.

## Environment adaptation (documented naming decisions)

This environment resolves FIRE **instantaneously** (hit-or-miss in the same
step); there is no travelling projectile sprite. The prompt's suggested
categories were mapped onto what this environment actually does:

| Prompt's suggested category | This benchmark's category |
|---|---|
| fire action produces projectile / projectile hits alien / alien disappears | `fire_hits_advances_target`, `fire_hits_clears_wave` |
| projectile advances | not applicable (no travelling projectile exists in this renderer) |
| cooldown activates | folded into the `fire_*` categories (cooldown flips the same step as fire) |
| cooldown decreases / cooldown clears | `cooldown_clears` (this environment's cooldown is a single-tick binary flag, not a multi-tick counter, so "decreases" and "clears" are the same event here) |

10 ordinary + 10 fault categories are implemented; see `dataset.py`'s
`ORDINARY_CATEGORIES` / `FAULT_CATEGORIES` for the exact list and the module
docstring/functions for what each one does and why.

## Known, documented limitations (read before trusting the numbers)

- **Direction-only faults are invisible at component granularity.**
  `tank_moves_wrong_direction` changes the tank component, just to the wrong
  column. Every system here reasons at the "did this named component change"
  granularity, not "did it move to the *correct* place" -- so this fault
  registers as `expected == observed == {"tank"}` for all three systems, and
  is correctly *not* flagged by any of them. This is a scope limitation of the
  chosen ground-truth granularity, not a system-specific weakness.
- **ZeroModel cannot know fire hit/miss.** It only sees frames + the action
  label, never `tank_x`/`target_x`/`cooldown`. It cannot form a crisp
  expectation for the alien band on FIRE (a hit and a miss look identical from
  the *action* alone), so `fire_no_projectile` (FIRE hits, but the alien wrongly
  stays put) is a structural blind spot: ZeroModel reports "conformant".
  `fire_no_cooldown` *is* caught, because cooldown-on-FIRE is asserted more
  aggressively (see the adapter module docstring for why that is fair).
- **Boundary-clamped tank moves produce a false alarm.** ZeroModel's own "tank
  must change on LEFT/RIGHT" expectation has no notion of screen edges, so
  `tank_remains_stationary_at_boundary` (a legitimate, non-faulty transition)
  is reported as `missing_expected_change`. This is exactly what the
  false-alarm-rate metric (6.5) is for, and it is reported, not hidden.
  - **P18C's per-episode demo is field-exact, not band-level.** Recurrence is
  computed on exact field ids, and the alien mark's column varies randomly
  across an episode's transitions, so most single 18-transition episodes do
  not accumulate enough exact-field recurrence to produce a candidate; a few
  do. This is a genuine property of P18C as written (field-level, not
  band-level, recurrence), not a bug in this demo.

## Stage 2: value-aware transition contracts

Stage 1 asks "did this named component change?" Stage 2
(`value_contracts.py`, `value_adapter.py`, `value_metrics.py`, `value_run.py`)
asks "did it change to the *correct value*?" -- direction, magnitude, cooldown
level, and one cross-field relation, all decoded from pixels via a *finer*
P4A field grid (1x1px, vs. stage 1's 4x1px) over the same existing
`build_transition_evidence_vpm`. No perception-package code changed; no new
P-stage was added. Stage 1's dataset, tests, and `run.py` are untouched --
`dataset.py` only gained new, additive category tables
(`VALUE_FAULT_CATEGORIES`) and a parallel `build_value_transition`/
`generate_value_episode`/`generate_value_split`.

Run it:

```bash
PYTHONPATH=examples python -m visual_transition_benchmark.value_run \
    --dev-episodes 40 --eval-episodes 120 \
    --output-dir artifacts/value_aware_transition_contracts
```

**Result**: resolves stage 1's documented wrong-direction blind spot
(`tank_moves_wrong_direction`), plus catches wrong-magnitude and
wrong-cooldown-value faults that stage 1 had no vocabulary for at all. It
remains honestly blind to target/alien *identity* faults
(`wrong_alien_disappears`, `two_aliens_disappear_instead_of_one`,
`fire_no_projectile`): those require the hidden alien queue, which no
non-privileged contract here can reconstruct from frames + action alone. See
`artifacts/value_aware_transition_contracts/value-benchmark-summary.md` for
the full breakdown.

## Directory layout

```
dataset.py            deterministic transition generator + fault injection
baselines.py           System A (pixel diff) + System B (privileged)
zeromodel_adapter.py    System C: P4A/P18A/P18B over static declared bands
discovery_demo.py       secondary P18C cohort-recurrence demonstration
metrics.py              section-6 metric definitions
render.py               diagnostic PNG panels + HTML index
report.py               aggregation into the required output files
run.py                  CLI entry point (stage 1)
value_contracts.py      stage 2: pixel-decoded typed values + contracts
value_adapter.py        stage 2: System D (value-aware ZeroModel)
value_metrics.py        stage 2: direction/magnitude/cooldown/target/relation metrics
value_run.py            stage 2 CLI entry point
tests/                  dataset / metrics / baselines / adapter / e2e-smoke tests
```
