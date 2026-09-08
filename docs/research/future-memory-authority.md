# Future memory authority: prediction accuracy is not decision authority

Follow-up to `future-memory-action-selection.md` (frozen; this document
reports the next bounded experiment, not a revision).

## The result that demanded this work

The first experiment established, on warehouse background shift, that
structured future memory improves choice (shared system 0.692 → 0.776,
net +0.038). But it also produced the more important finding: the ridge
model predicts *better* futures (MAE 0.010–0.013 vs empirical 0.019–0.022)
yet caused harm 0.50 at coverage 0.55 under arcade appearance shift.

```text
prediction accuracy
        ≠
decision authority
```

Remembering something is not the same as allowing that memory to command
enactment. Two defects shared the blame:

1. The coupling vetoed on contradiction *before* consulting confidence:
   `low_confidence_threshold` described the projection but did not limit
   its authority to veto.
2. OOD used distance-to-action-centroid, which misdescribes possibly
   multimodal support (the same weakness already noted in Visual Sign
   Reader work, where global calibration was judged too conservative).

## What was built (one bounded PR, perception only)

**Memory Validity VPM** (`memory_authority.py`): `FutureMemoryValidityDTO`
accumulates verification outcomes per (transition model, action) —
counts, a bounded recent-outcome window (16), summed projection /
direction / changed-field error magnitudes — into content-addressed
evidence. `record_verification_event` folds each
`FutureTransitionVerificationDTO` into the next state; nothing is refit.

**Authority derivation**: `assess_memory_authority` maps validity evidence
plus the live projection to SUPPORT_ONLY / MAY_VETO / STALE / OOD. Cold
memory with a supported confident projection stays permissive (preserving
genuine corrections); accumulated mismatch demotes to STALE no matter how
confident the projection claims; low confidence demotes to SUPPORT_ONLY.
The gate vetoes a contradiction only under MAY_VETO — otherwise it
annotates. Every candidate carries its `memory_authority` level and every
reason chain names it.

**Local support** (`transition_model.py`, `transition_projection.py`):
per-action typical nearest-neighbour spacing is compiled at fit time and
stored frozen; OOD and confidence proximity now use distance-to-local-
support instead of distance-to-centroid, for both empirical and ridge
models (ridge scans its stored training examples).

**Integrity fix**: both model DTOs now carry `canonical_payload()` and
their constructors reject any identity that does not bind the exact
runtime payload — coefficients, residuals, centroids, spreads, typical
spacings, per-example pixel digests. Tampering with a single centroid or
coefficient bit raises at construction (tested).

**Verification signal enriched** (`transition_verification.py`): reports
now carry mean absolute error, direction error rate, and changed-field
error rate alongside statuses, feeding the validity accumulator.

No governance, no promotion, no mined expectations, no neural machinery.
`predict_action_with_future_memory` keeps its four positional arguments;
declaration context travels as one `DeclarationScopeDTO` (10 params total,
at the quality limit, not over it).

## Falsifiable check: ridge + background shift

Prediction BEFORE: harm 0.50, coverage 0.55, net −0.497.
Result AFTER (seeds 0–2): **harm 0.000, coverage 1.000, net +0.003.**

Mechanism telemetry (per query, arcade bg-shift, ridge): vetoes 0.000,
STALE flags 1.61, MAY_VETO flags 0.000. Two protections compose: (1) the
local-scale proximity demotes shifted projections below veto confidence
immediately (no cold-start vetoes), and (2) accumulating mismatches mark
the memory STALE (shared-system may-veto rate falls 0.58 → 0.08 across
per-episode halves while its harm falls 0.021 → 0.000).

Genuine corrections survive: warehouse background shift keeps D net
+0.034 (harm 0.009) and B net +0.004 (harm 0.004); halves show early
gains retained (+0.063 → +0.016 for D). Prediction degrades while
decisions stay protected: ridge bg-shift verification mismatches stay
high (279) with zero decision harm — the dissociation is now crisp.

## Honest limits

- Warehouse goal-change still shows B/C harm 0.05 at coverage 0.95:
  under genuine task change the gate *abstains* (OOD) rather than
  guessing. Authority correctly does not suppress OOD caution; whether
  abstention beats guessing there is an open policy question, not a bug.
- Noise splits report declared violations of exact-zero stable
  expectations (correct behavior of the loop, but it masks projection
  comparison; match-given-conformant is reported alongside).
- Halves compare per-episode first/second steps pooled across episodes
  (same episodes both sides), not across different random starts.
- Effect sizes remain small; the shared predictor still collapses under
  task change (0.031) where the gate can only decline to harm.

## Verdict answers

Q resemblance to the proposal: the veto now requires MAY_VETO; STALE and
low-confidence memories annotate only — exactly the requested
contradiction → trust-check → veto-or-annotate order. `model_id` binds
the exact runtime payload. OOD is local (typical spacing, neighbour
distance, agreement via dispersion, residuals) for both model kinds.

Stop/go for THIS loop: **GO.** The known failure is fixed (0.50 → 0.000
harm) without destroying genuine corrections, by the predicted
mechanism, with telemetry to prove it. Next: confidence-gated
contradiction thresholds mined from dev (not hand-set), local-density
abstention policy for genuine task change, then real-pipeline tasks.
Still not: promotion lifecycle, larger models, causal/safety claims.

Command (from `examples/`):
`PYTHONPATH=../packages/core/src ../venv/Scripts/python -m
visual_transition_benchmark.future_memory_benchmark --seeds 0,1,2
--train-episodes 24 --eval-episodes 10 --output-dir
../artifacts/future_memory_benchmark/authority2`
