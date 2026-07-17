# Phase 2 option set: toward a working ZeroModel visual reader

**Status:** design options after Phase 1  
**Decision gate:** complete the evidence-closure PR and failure atlas first  
**Current capability statement:** ZeroModel does not yet have a validated visual AI

---

## 1. Why continue

Phase 1 did not show that visual addressing is impossible.

It showed that one attractive shortcut did not work:

```text
whole image
    → global DINOv2 CLS vector
    → nearest policy row
```

The provider-neutral contracts, exact policy lookup, benchmark harness, and
negative result now make a second attempt much more disciplined.

The next attempt should preserve what works and replace only the observation
mechanism.

---

## 2. Locked foundation

The following components are retained:

```text
VPM policy artifact
exact symbolic row encoding
VPMPolicyLookup
ImageObservation
VisualAddressContract
VisualAddressDecision
VisualAddressProvider
encoder / preprocessing / calibration identities
family-held-out benchmark
information-theoretic controls
first-class rejection
```

The following Phase 1 provider is not promoted:

```text
global DINOv2 CLS row retrieval
```

The following provider remains unresolved:

```text
calibrated normalized pixels
```

---

## 3. Three candidate routes

### Route A — calibrated pixel Level 1

Hypothesis:

> In a fixed-geometry bounded scene, normalized pixels have a high-precision
> operating region that provides useful tolerant addressing with explicit
> abstention.

Required experiment:

- create independent rejection-calibration families;
- predeclare a FAR target or risk-weighted loss;
- choose score and margin thresholds on calibration only;
- evaluate once on untouched benign and invalid families;
- report coverage, exact-row precision, action precision, FAR, and FRR.

Continue when:

- exact-row precision is high enough for the intended use;
- FAR meets the declared target;
- refusal cost is acceptable;
- pixels remain simpler than learned alternatives.

Stop when:

- no threshold provides a useful precision/coverage region;
- small environmental changes destroy calibration;
- example count grows toward exhaustive memorization without value.

### Route B — local learned evidence

Hypothesis:

> The failure is caused primarily by global aggregation; spatial patch evidence
> can preserve policy-critical object existence and position.

One bounded probe:

- CLS token;
- mean patch token;
- spatial patch bins;
- direct patch-token retrieval;
- one intermediate layer if inexpensive.

Continue when:

- exact top-1 row ranking improves materially over B or complements B;
- critical intervention separation improves;
- the gain survives held-out families;
- evidence remains spatially inspectable.

Stop after one bounded negative probe.

Do not expand into broad model shopping.

### Route C — factorized Visual State Compiler

Hypothesis:

> Explicit extraction of policy-critical factors produces safer and more
> diagnosable state reconstruction than direct whole-row retrieval.

Proposed flow:

```text
ImageObservation
    ↓
factor providers
    ↓
EvidenceBundle
    ↓
TypedObservedState
    ↓
deterministic state encoder
    ↓
exact VPM policy row
```

Arcade factors:

```text
tank_present
tank_count
tank_position
target_present
target_position
cooldown_state
frame_structurally_valid
```

Mandatory baselines:

1. direct engine instrumentation;
2. deterministic connected components / geometry;
3. local template matching;
4. small task-specific factor classifiers;
5. patch-token factor probes;
6. compact multi-head factor model.

The learned component is used only where deterministic extraction fails.

---

## 4. What “visual AI” would mean here

A successful bounded ZeroModel visual AI would not need to claim general image
understanding.

It would need to demonstrate:

1. visual variation not enumerated in the policy artifact;
2. reliable extraction of visible state;
3. explicit uncertainty and refusal;
4. rejection of missing or contradictory evidence;
5. exact delegation into the compiled policy;
6. replayable evidence and provider identity;
7. performance beyond direct pixel codewords;
8. value over direct instrumentation or a conventional detector/log stack.

That is a real visual system, even if it is bounded.

---

## 5. Phase 2 benchmark layers

| Layer | Measurement |
|---|---|
| Object presence | precision, recall, abstention |
| Object count | exact count |
| Position | exact policy cell |
| Cooldown/state flag | exact classification |
| Missing evidence | rejection precision and recall |
| Structural validity | invalid combination detection |
| Complete state | exact state reconstruction |
| Policy row | exact address |
| Policy action | action correctness |
| Safety | FAR, FRR, severity-weighted loss |
| Calibration | reliability and family transfer |
| Evidence | region correctness and trace completeness |
| Cost | runtime, memory, examples, calibration effort |

The benchmark must retain “correct action from wrong state” as a distinct
partial-success category.

---

## 6. Recommended order

```text
1. merge evidence closure
2. attach original raw v1 report
3. generate full failure atlas
4. run independent B calibration experiment
5. run bounded patch-token locality probe
6. choose B, local evidence, factorized compiler, or no visual path
7. perform governance-parity audit before expanding deployment contracts
8. move to a fixed-camera bounded environment
```

---

## 7. First realistic deployment question

Before building more perception, name one environment where:

- the state is bounded;
- direct telemetry is unavailable, untrusted, or independently verified by
  vision;
- a camera or screenshot is the natural observation surface;
- false acceptance has a meaningful cost;
- exact policy evidence is useful.

Candidate categories:

- fixed industrial panel;
- legacy UI automation;
- game or simulator without internal state access;
- visual verification of a separate telemetry channel;
- bounded accessibility interface;
- fixed-camera inventory or machine-state panel.

The environment should be chosen before claiming deployment value.

---

## 8. Bottom line

Yes, ZeroModel should try another visual approach.

It should not erase the failed attempt or immediately replace DINOv2 with a
larger foundation model.

The disciplined route is:

> lock the Phase 1 evidence, test whether calibrated pixels already provide a
> useful Level 1, probe local learned evidence once, and then build a factorized
> state compiler only if the measurements justify it.
