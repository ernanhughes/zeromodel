# External review adjudication and revised visual research sequence

**Status:** adopted research correction  
**Date:** 17 July 2026  
**Scope:** Phase 1 result preservation, threshold interpretation, failure-atlas requirements, and the route toward a working visual observation provider

---

## 1. Decision

The external review is substantially accepted.

It does not reverse the Phase 1 negative result. It makes the result more
precise and changes the order of the next work.

The revised sequence is:

```text
preserve the original run
    ↓
repair provenance and reporting
    ↓
measure ranking quality separately from calibration
    ↓
measure full score/coverage/FAR/FRR curves
    ↓
test whether normalized pixels provide a useful Level 1
    ↓
run one bounded patch-token locality probe
    ↓
choose the next visual architecture
```

The project should not jump directly from the failed global DINOv2 reader to a
large Visual State Compiler implementation.

---

## 2. What remains strongly negative

The current global DINOv2 CLS retrieval path is not promoted.

At the measured permissive operating point, C and D:

- recover the exact policy row on only about 31% of accepted benign cases;
- generate hundreds of accepted conflicting-action decisions;
- accept a large majority of distinguishable rejection opportunities;
- provide no demonstrated governance or complexity advantage over normalized
  pixels.

D's 74.78% benign action accuracy is not sufficient evidence of governed
addressing because 597 of its 1,005 correct actions came from the wrong policy
row.

The linear probe G remains elimination-grade at the measured point:

- 59.45% benign action accuracy;
- 28.79% exact benign row accuracy;
- 100% FAR;
- 521 conflicting-action errors.

Operating curves may characterize these failures, but the current evidence does
not justify further development of G.

---

## 3. What remains open

System B must be treated differently from C, D, and G.

At the permissive measured point, B has:

```text
benign action accuracy:           73.44%
exact benign row accuracy:        62.50%
exact row when accepted:          83.33%
correct action when accepted:     97.92%
false-acceptance rate:            51.21%
false-rejection rate:             25.00%
```

This is not deployment-ready.

It is also not the same failure as DINOv2. B's ranking is often correct when it
accepts, while its refusal boundary is too permissive.

The correct present wording is:

> Normalized pixels failed the rejection requirement at the maximally
> permissive measured operating point. A useful high-precision,
> lower-coverage operating region remains unresolved.

A complete B operating curve is now the highest-value diagnostic.

---

## 4. Threshold-dependent and ranking-dependent evidence

The original report mixed two questions:

1. Is the nearest candidate correct?
2. Does calibration accept the observation?

These must be measured separately.

### Ranking metrics

Record before rejection:

```text
top1_row_id
top1_action_id
top1_correct_row_count
top1_correct_action_count
correct_row_rank
nearest_score
correct_row_score
best_conflicting_action_score
```

These metrics ask whether the representation and matcher order candidates
correctly.

### Calibration metrics

Record after applying the acceptance rule:

```text
coverage
accepted-row precision
accepted-action precision
FAR
FRR
correct rejection
```

These metrics ask whether the system has a useful operating region.

A threshold cannot change one observation's nearest-neighbour ranking. It can,
however, prune low-confidence observations and raise precision among the
remaining accepted subset. Therefore exact-row-when-accepted is not fully
threshold-independent.

---

## 5. Failure atlas requirements

The new `zeromodel.visual_analysis` module and
`examples/analyze_visual_address_report.py` provide the first executable atlas
scaffold.

The atlas must include:

### 5.1 Score-conditioned operating curves

For each system:

- coverage;
- accepted exact-row precision;
- accepted action precision;
- exact-row recall;
- action recall;
- FAR;
- FRR;
- observation-level Wilson intervals.

The current evaluation set may be used for exploratory curves only.

A threshold selected from those curves requires:

- an independent rejection-calibration split;
- an untouched final evaluation split;
- a predeclared target or loss.

### 5.2 Paired B-versus-D outcomes

For every benign observation, record whether B and D get the top-1:

- row correct;
- action correct.

The off-diagonal counts support a paired McNemar comparison. Research reporting
should additionally use state- and family-clustered uncertainty because
observations generated from the same state and corruption family are not fully
independent.

### 5.3 Translation-family analysis

Test the mechanistic hypothesis that:

```text
tiny 16×28 geometry
    → letterbox and upscale
    → interpolation and processor normalization
    → global CLS aggregation
    → local position and existence evidence weakened
```

Required evidence:

- original frame;
- letterboxed frame;
- final processed tensor visualization;
- source-state distance between expected and retrieved rows;
- tank-position and target-position confusion;
- action-equivalent versus conflicting-action confusion;
- B-success/D-failure and D-success/B-failure examples.

### 5.4 Bounded patch-token probe

Compare:

- CLS token;
- mean patch token;
- spatially pooled patch bins;
- direct patch-token retrieval;
- optionally one intermediate layer.

This is a diagnostic branch, not automatic permission to revive global learned
row retrieval.

A one-day negative result should end it. A strong positive result may justify
using local learned evidence inside a factorized state provider.

---

## 6. Getting closer to visual AI

ZeroModel does not currently have a validated visual AI.

It has:

- an exact canonical visual codeword reader;
- an approximate pixel reader with unresolved calibration potential;
- failed global frozen-embedding baselines;
- a provider-neutral observation-address contract;
- benchmark machinery capable of rejecting a visual hypothesis.

That is a strong foundation for another attempt.

The next attempt should proceed in levels.

### Level 0 — exact canonical reader

Use when observations obey an exact visual protocol.

Properties:

- deterministic;
- exact;
- inspectable;
- strict refusal;
- no learned inference.

### Level 1 — calibrated normalized pixels

Use when fixed geometry and appearance are stable enough for pixel similarity.

Research question:

> Can B reach a declared high-precision operating point with an acceptable and
> explicitly priced refusal rate?

This may produce a useful visual reader without a learned model.

### Level 2 — factorized visual evidence

Use when appearance variation exceeds what Level 1 can handle.

Proposed flow:

```text
ImageObservation
    ↓
local evidence providers
    ↓
EvidenceBundle
    ↓
TypedObservedState
    ↓
exact state encoder
    ↓
VPMPolicyLookup
```

Possible factors:

```text
tank_present
tank_count
tank_position
target_present
target_position
cooldown_state
frame_structurally_valid
```

The visual model, when used, should solve local evidence problems rather than
jump directly from a whole image to one of 112 policy rows.

### Level 3 — temporal or multisensor evidence

Use only for cases that are impossible from one frame, such as a removed target
whose resulting pixels are identical to a legitimate no-target state.

---

## 7. The research discovery

The strongest current hypothesis is:

> In systems where small visible distinctions determine exact governed state
> identity, representation invariance can become a governance failure rather
> than a robustness benefit.

This is more precise than the slogan "invariance is the enemy of identity."

The measured result supports the hypothesis in one tiny bounded fixture. It
does not yet establish a general law.

Replication is required in:

- a second synthetic governed environment;
- a UI or fixed-camera bounded environment;
- a task-specific or factorized baseline;
- a governance-parity comparison.

---

## 8. Governance parity is existential

Before expanding the artifact chain, compare it with a conventional minimum:

### ZeroModel stack

```text
policy artifact
encoder manifest
representation identity
prototype bindings
calibration artifact
deployment binding
decision trace
benchmark report
```

### Lightweight stack

```text
policy/config SHA-256
model SHA-256
preprocessing config
threshold config
Git commit
append-only JSONL decision record
```

Predeclare incident questions:

- Which policy and threshold were active?
- Which decisions were affected by calibration X?
- Can the candidate set be reconstructed?
- Can a rejection be replayed?
- Can tampering be detected?
- Which representation and model produced the decision?
- How long does each investigation take?

If the lightweight stack reaches audit parity at materially lower complexity,
ZeroModel should reduce the governance architecture.

---

## 9. Immediate PR boundary

This evidence-closure PR includes:

- the corrected PowerShell runner;
- first-class benign and top-1 metrics;
- observation-level Wilson intervals;
- top-1 trace fields independent of acceptance;
- per-row calibration counts;
- explicit no-conflicting-action semantics;
- execution-scoped float representation identity wording;
- the CI summary serialization fix;
- exploratory operating-curve, family, and paired-outcome tooling;
- the v1 aggregate result and provenance limitation record;
- the revised claims ladder and visual claim wording.

It intentionally does not include:

- a new trained model;
- threshold selection on the current test set;
- a claim that B is safe;
- a claim that patch tokens work;
- a full Visual State Compiler;
- physical deployment;
- open-world perception.

Those are next experiments, not closure fixes.
