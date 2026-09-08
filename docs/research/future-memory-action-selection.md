# Structured future memory for action selection (DreamWAM/JEPA-WAM experiment)

## Motivation

DreamWAM: the future should be represented in action-relevant structured form,
not as RGB pixels. ZeroModel translation: an `ExpectedTransitionVPMDTO`
records per field what changes, where, in what direction, by approximately
how much, how stably (dispersion), and with how much historical support —
reusing P18A measurement semantics, never synthesizing a future frame. The
PNG it carries renders expected absolute change per field: a visualization
*of* the structured expectation for inspectability.

JEPA-WAM: future-state prediction must influence the representation used for
action selection rather than existing as a separate post-action diagnostic.
ZeroModel translation: `predict_action_with_future_memory` projects a
remembered consequence for each bounded candidate *before* enactment and lets
it support / constrain / contradict / flag-OOD the candidacy — while the
baseline policy score stays the sole utility source (selection order among
survivors is always baseline order; predictability never manufactures
utility).

ZeroModel interpretation: `ExpectedTransitionVPM = compiled remembered
consequence`. Observe (Source VPM) → Remember (policy memory + evidence +
expected-transition model) → Choose (world-action coupling) → Enact
(external caller) → Strengthen/Return (expected-vs-observed verification).
No persistent learning: evidence is recorded, never auto-promoted.

## Claim boundary

Valid claim: *ZeroModel can compile action-conditioned structured transition
evidence from historical before/action/after observations and use that
remembered future as bounded evidence during action selection.*

Not claimed: general world modeling, causal understanding, realistic future
simulation, open-world planning, robotics capability, autonomous safe
control. Language used throughout: projected / predicted / expected /
supported / observed / conformant / contradicted / out-of-distribution.

## Architecture

Seam: `zeromodel.perception` owns the lifecycle; no new package. Six small
additive modules plus one behavior-preserving refactor:

- `expected_transition.py` — `ExpectedTransitionFieldDTO`,
  `ExpectedTransitionVPMDTO` (statuses: supported / insufficient_examples /
  out_of_distribution / ambiguous_future / unsupported_action; confidence is
  capped for every non-supported status, never silently confident).
- `transition_model.py` — `fit_action_conditioned_transition_model`
  (empirical kNN over P18A evidence built per eligible transition; only
  interactions with an authoritative `next_source_vpm_id` are eligible) and
  `fit_compiled_transition_model` (per-action closed-form ridge,
  before-field-means → delta-field-means, NumPy only).
- `transition_projection.py` — pure `project_expected_transition` (no
  fitting; deterministic content-derived identity; projected level anchored
  at the present source: after = before + expected change).
- `transition_conformance.py` — extracted the pure P18B decision tree into
  `_decide_transition_status`, reused verbatim by observed classification and
  by pre-enactment expected-conformance checks (no duplicated logic).
- `world_action.py` — `WorldActionPolicyDTO`,
  `check_expected_conformance`, `predict_action_with_future_memory`
  (preserves the whole `BaselinePredictionDTO`; per-action expectation
  scoping lives with the caller, mirroring the arcade adapter's
  `EXPECTATIONS_BY_ACTION`; optional `baseline_override` lets a same-family
  ranking such as the relevance-weighted predictor drive System C
  coherently).
- `transition_verification.py` — `verify_expected_transition` comparing a
  remembered future against a `VisualTransitionAnalysisDTO` (statuses:
  confirmed / confirmed_with_unexpected_change / future_projection_mismatch
  / declared_expectation_violation / insufficient_evidence; declared
  violations take precedence as authoritative).
- `shared_relevance.py` — `fit_shared_field_relevance` reuses
  `estimate_field_relevance` for the action term and adds eta-squared of
  field deltas for the future term (0.3/0.7); `predict_relevance_weighted_action`
  reuses the P3 memory and vote/reject contract with relevance distances.

Reused unmodified: P1 source/action encoding, P2 ledger + manifests, P3
predictor, P4A schemas, P6 annotations, P18A evidence builder, P18B report
machinery, P18 analysis composition, `VPMMetadata`-style `sha256:` identity
discipline, arcade adapter schema/annotations/expectations. Public API
exposed through `zeromodel.perception.__init__` with a public-surface test.

## Evaluation protocol

`examples/visual_transition_benchmark/future_memory_benchmark.py` drives the
REAL `TinyArcadeShooter` / warehouse engines and renderers. Train episodes
are disjoint from dev/clean/nuisance/transition-change episodes (episode
ids asserted disjoint by construction); all models fit on the train
manifest only (`training_split="all"` — each manifest IS its split).
Parameters fixed a priori (k=16, min_support=10, OOD×3, ambiguity 0.2,
change_threshold=8 matching the adapter, ridge α=1.0, verification
tol 0.02/eps 0.01, nuisance bg-shift +30 on both-frame background,
sub-threshold noise ±3, changed wave `(6,0,5,1)`, relocated warehouse goal).
Nothing is selected on dev/eval. Seeds {0,1,2} averaged; 24 train episodes,
10 eval episodes per split. Arcade labels reproduce the repo's own compiled
policy table (asserted cell-by-cell in harness tests); warehouse labels are
greedy navigation (harness task definition).

Command (from `examples/`, core payload is not an installed distribution):

```text
PYTHONPATH=../packages/core/src ../venv/Scripts/python -m \
  visual_transition_benchmark.future_memory_benchmark --seeds 0,1,2 \
  --train-episodes 24 --eval-episodes 10
```

Results: `artifacts/future_memory_benchmark/seeds012/future-memory-results.json`
(evaluated commit `daf4bba` plus additive files).

## Results (mean over seeds 0,1,2)

### Warehouse (informative selection domain, 10 actions)

| System | Clean acc. | Bg-shift acc. | Coverage | Gain | Harm | Net | Future MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A baseline | 0.735 | 0.692 | 1.00 | n/a | n/a | n/a | n/a |
| B +empirical gate | 0.739 | 0.696 | 0.99/0.94 | 0.004/0.009 | 0.000/0.004 | +0.004/+0.004 | 0.019/0.022 |
| C +compiled ridge | 0.739 | 0.696 | 0.99/0.94 | 0.004/0.009 | 0.000/0.004 | +0.004/+0.004 | 0.010/0.013 |
| D +shared repr | 0.744→0.753 | 0.738→0.776 | 0.99/0.96 | 0.019/0.047 | 0.009/0.009 | +0.010/+0.038 | 0.021/0.026 |

### Arcade (baseline at ceiling 1.0; tests prediction + verification)

| System | Clean | Bg-shift (coverage) | Net | Harm | Future MAE |
|---|---:|---:|---:|---:|---:|
| A baseline | 1.000 | 0.997 | n/a | n/a | n/a |
| B +empirical gate | 1.000 | 1.000 (1.00) | +0.003 | 0.000 | 0.008 |
| C +compiled ridge | 1.000 | 0.500 (0.55) | −0.497 | 0.500 | 0.003–0.008 |
| D +shared repr | 1.000 | 0.990 (0.99) | −0.010 | 0.010 | 0.019–0.021 |

Future quality: empirical projection match 0.98 arcade / 0.86 warehouse
clean; wave-change drops arcade match 0.98→0.87 with mismatches 9→35;
warehouse goal-move drops match 0.86→0.69. Ridge has the best levels
(MAE 0.003–0.013) but the worst changed-field F1 (0.49–0.58 vs empirical
0.50–0.83: binary fractions lose calibration) and direction accuracy.
Verification loop: clean mostly confirmed; stale-memory splits raise
mismatches; noise splits (correctly) report declared violations of the
adapter's exact-zero stable expectations.

## Research verdict

1. **Useful structured futures? Yes.** Supported projections reach
   MAE 0.003–0.02 with match rates 0.86–0.98, and the verification loop
   confirms clean transitions while flagging stale memory (arcade wave
   change: mismatches 9→35).
2. **Does future prediction improve action choice? Modestly, conditionally.**
   Warehouse: D net +0.010 clean / +0.038 bg-shift / +0.019 noise; B/C net
   +0.004; harm ≤0.009. Arcade at ceiling: only B's single correction
   (+0.003). Most baseline errors are wrong-action/right-shape confusions
   no plausibility gate can catch (§8 limit, confirmed empirically).
3. **Robustness? Partial.** D coupled beats the clean baseline under bg
   shift (0.776 vs 0.735); B never degrades. But C catastrophically vetoes
   correct actions under bg shift (arcade net −0.497, coverage 0.55):
   ridge extrapolates appearance shifts into contradictions. Empirical
   locality is the robust coupling; global linear maps are not.
4. **Where does future memory hurt?** (a) Compiled gate + appearance shift
   (harm 0.50) — do not gate on extrapolated means without confidence
   gating. (b) Any gate + stale memory (warehouse goal move: B/C harm
   0.05, coverage 0.95) — memory staleness must gate the gate; the
   verification loop already detects it (mismatch surge). Otherwise harm
   is 0–0.009 with coverage 0.93–1.0.
5. **Shared better than separate gate? Partly.** D wins warehouse selection
   (+0.038 net under shift, best clean 0.753) but its future MAE is worse
   than B/C and its predictor collapses under task change (0.031, rescued
   to 0.115 by the uniform gate inside). Shared representation helps
   decisions, hurts projections: keep the gate's projector uniform or
   relevance-weighted per-side, not blindly shared.
6. **Build next:** mined (not hand-authored) per-action expectations with
   human review; confidence-gated contradiction vetoes; local-density OOD
   for honest abstention; real-pipeline task validation (finance/trading
   benchmarks) instead of gridworlds.
7. **Do NOT build:** promotion/governance lifecycle for transition models,
   larger neural/video machinery, causal/safety claims. The effect sizes
   are small, the compiled gate is unsafe under shift, and the shared
   representation result argues against deeper coupling without new
   evidence.

Stop/go: **CONDITIONAL GO.** Net correction > 0 at useful coverage with
near-zero harm is demonstrated (empirical gate everywhere, shared system
on warehouse), future prediction is solid, and the verification loop
works — but the compiled gate must be barred from gating under
appearance shift until contradiction vetoes are confidence-gated, and
nothing here justifies promotion-lifecycle integration.
