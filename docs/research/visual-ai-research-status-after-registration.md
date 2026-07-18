# Visual AI Research Status After Bounded Registration

**Status:** Phase B Stage 1 completed
**Date:** July 18, 2026
**Measured system:** `R1 / registered_local_normalized_pixels`
**Measured outcome:** `C / bounded_registration_insufficient`
**Next experiment:** `translation_equivariant_template_correlation`

## Executive summary

ZeroModel has not yet demonstrated a useful governed visual-address reader, but the Phase B Stage 1 result materially advances the visual-AI research programme.

Bounded deterministic registration improved exact policy-row retrieval from `75%` to `87.5%` and improved action retrieval from `96.875%` to `98.4375%`. On the deliberately held-out two-pixel translation family, registration recovered every expected policy row and action correctly at raw top-1.

This demonstrates that a substantial part of the earlier visual-address failure came from spatial misalignment rather than an absence of usable visual information.

However, the improved ranking did not produce a useful governed operating point under the declared rejection constraints. The selected calibration preserved zero observed distinguishable false accepts and zero accepted conflicting-action errors, but accepted none of the `1,344` benign final observations.

The present result therefore separates two problems that had previously been entangled:

1. identifying the most likely policy state from an image;
2. proving that the selected state is sufficiently well-supported to act upon safely.

The first problem is becoming tractable on the bounded synthetic fixture. The second remains unresolved.

---

## Research question

Phase B Stage 1 asked:

> Can bounded deterministic integer-pixel registration recover a useful accepted operating point for local normalized-pixel retrieval while preserving zero observed distinguishable false accepts and zero accepted conflicting-action errors on the declared fixtures?

The experiment deliberately avoided learned models.

It used:

* native-resolution images;
* exhaustive bounded integer translation;
* no interpolation;
* no image resizing;
* no edge wrapping;
* valid-overlap-only comparison;
* deterministic candidate ordering;
* calibration-only operating-point selection;
* frozen final evaluation;
* the existing governed visual-address evaluation framework.

The registration search was bounded to:

```text
dx ∈ [-3, 3]
dy ∈ [-3, 3]
```

The declared final translation family used unseen shifts of magnitude two pixels.

---

## Headline results

| Metric                             | Frozen System B | Registered R1 |                                  Change |
| ---------------------------------- | --------------: | ------------: | --------------------------------------: |
| Raw top-1 exact-row correctness    |   `1008 / 1344` | `1176 / 1344` |                                  `+168` |
| Raw top-1 exact-row accuracy       |         `75.0%` |       `87.5%` |                          `+12.5 points` |
| Raw top-1 action correctness       |   `1302 / 1344` | `1323 / 1344` |                                   `+21` |
| Raw top-1 action accuracy          |       `96.875%` |    `98.4375%` |                        `+1.5625 points` |
| Accepted benign observations       |      `0 / 1344` |    `0 / 1344` |                      no useful coverage |
| Distinguishable false accepts      |       `0 / 248` |     `0 / 248` |                    safety gate retained |
| Accepted conflicting-action errors |             `0` |           `0` |                    safety gate retained |
| False rejects                      |   `1344 / 1344` | `1344 / 1344` |                               unchanged |
| Outcome                            |             `C` |           `C` | ranking improved; governance unresolved |

The selected R1 operating point used:

```text
selected quantile: 0.0
selected threshold: 0.07306751140001405
selected ambiguity margin: 0.6782577226161598
```

Its frozen identities include:

```text
selection digest:
881d63866587aec8d87d4d3419431ccd195f6fe57828e9f70580ebea1e0337f4

calibration digest:
b3efc2a4687ba97a5b899961a8d3b53b91f1e0f2c73afc7bde92a1bb8e805265

trace digest:
c1416850155cb464a70b55fd8856a240ea4d86b6ae3f85afb1948899469dbfec
```

---

## Translation-family result

The strongest positive result occurred on the held-out two-pixel translation family.

Before registration:

```text
exact-row correctness:
224 / 336

action correctness:
322 / 336
```

After bounded registration:

```text
exact-row correctness:
336 / 336

action correctness:
336 / 336
```

This is important because the final translation family was not used to select the registration result.

The experiment therefore supports the bounded claim:

> Deterministic integer registration can generalize to the declared unseen bounded translations and completely repair that translation family at the raw ranking layer.

It does not establish general translation invariance beyond the declared bounds or fixture.

---

## What the experiment established

### 1. The visual state is recoverable from the pixels

The earlier result left open the possibility that normalized pixel observations did not contain enough information to distinguish the relevant policy states.

The registered result weakens that explanation considerably.

An exact-row accuracy of `87.5%` and action accuracy of approximately `98.4%` show that the images contain substantial policy-address information.

The remaining problem is not simply that the visual state is invisible.

### 2. Spatial alignment was a genuine source of error

A simple deterministic alignment mechanism produced a large improvement without:

* neural training;
* additional supervision;
* semantic embeddings;
* final-evaluation tuning;
* larger models;
* stochastic search.

This demonstrates that locality and coordinate alignment are important parts of the visual-address problem.

The result supports continuing to investigate local evidence before promoting a learned visual model.

### 3. Correct action retrieval is easier than exact-state retrieval

R1 selected the correct action for `1323` of `1344` benign final observations, but selected the exact policy row for only `1176`.

This gap matters.

Multiple visually similar rows may imply the same action. A system can therefore appear behaviourally correct while still misunderstanding the exact state.

ZeroModel requires the distinction to remain visible because:

* policy-row identity carries provenance;
* future rows with similar appearances may require different actions;
* action equivalence does not prove state equivalence;
* governance must identify conflicting-action near-matches explicitly.

### 4. Ranking and governed acceptance are different capabilities

R1 often chose the correct state but could not prove that its choice was safe enough to accept.

The selected operating point achieved:

```text
false accepts:
0 / 248

accepted conflicting-action errors:
0
```

but only by rejecting all benign final observations.

This establishes an important architectural distinction:

> A visual reader may rank the correct answer highly without possessing sufficient evidence to govern the decision.

ZeroModel should not collapse these into one metric.

### 5. The current bottleneck is evidence separation

The current system compares registered global pixel regions and derives confidence from distance and conflicting-action margin.

That mechanism does not produce a useful separation between:

* correct exact-row matches;
* visually close wrong-row matches;
* conflicting-action near-matches;
* distinguishable out-of-distribution controls.

The next stage must improve the evidence representation rather than merely improve top-1 guessing.

---

## What the experiment did not establish

The Stage 1 result does not prove that:

* ZeroModel has solved visual understanding;
* R1 is a deployable visual reader;
* the system has a nontrivial safe acceptance region;
* bounded registration fails under every possible calibration search;
* the current zero observed false-accept result generalizes outside the declared fixture;
* the method is robust to arbitrary translation;
* the method is robust to scale, rotation, perspective, blur or occlusion;
* the method works on physical camera captures;
* the method operates in real time;
* a learned model is unnecessary;
* a complete Visual State Compiler has been implemented.

The phrase “zero false accepts” must not be used without qualification.

The supported wording is:

> Zero observed distinguishable false accepts on the declared calibration and final fixtures at the selected operating point.

---

## Calibration-search qualification

The current Stage 1 candidate search couples the distance threshold and ambiguity margin through one shared quantile.

This tests a predeclared one-dimensional operating curve rather than every combination in the two-dimensional distance-and-margin space.

The result therefore most directly establishes:

> No useful governed operating point was found on the declared coupled-quantile calibration curve.

A future calibration refinement may test a Cartesian grid of independent:

```text
distance quantiles
×
ambiguity-margin quantiles
```

That refinement could strengthen or revise the Stage 1 conclusion without changing the underlying registration measurements.

Until that search is performed, “bounded registration is insufficient” should be interpreted as referring to the declared Stage 1 provider and calibration protocol, not every conceivable registered-pixel rejection policy.

---

## Position on the visual-AI proof ladder

### Established on the declared synthetic fixture

* Pixel observations contain meaningful policy-state information.
* Deterministic images can be linked to exact policy rows.
* Raw visual retrieval contains very strong action information.
* Bounded integer registration materially improves exact-row retrieval.
* The unseen two-pixel translation family is fully recovered at top-1.
* Evaluation can remain isolated between calibration and final splits.
* Results can be serialized, hashed, reconstructed and audited.
* The system can preserve its safety gates by rejecting ambiguous observations.

### Not yet established

* Nontrivial accepted benign coverage under the safety gates.
* Reliable separation of exact matches from dangerous near-matches.
* Robustness outside the synthetic renderer.
* Physical camera performance.
* Real-world lighting and display variation.
* Scale or rotational invariance.
* Object and geometry understanding.
* Real-time runtime viability.
* General-purpose visual intelligence.
* A complete factorized visual state compiler.

---

## Current interpretation of the ZeroModel visual thesis

The visual thesis is no longer purely speculative.

The research now supports this bounded statement:

> A ZeroModel policy can be addressed from pixels with high raw action accuracy and materially improved exact-row accuracy when deterministic local alignment is introduced. The remaining unsolved problem is producing sufficiently discriminative evidence to govern acceptance safely.

This is progress because it replaces a broad question—

> Can ZeroModel read visual state at all?

—with a narrower and more actionable question—

> What local evidence is needed to prove that the selected visual state is the correct and safe policy address?

---

## Why the result matters

A conventional classification evaluation might focus on the approximately `98.4%` action accuracy and describe the system as nearly solved.

ZeroModel applies a stricter standard.

It asks whether the system can demonstrate:

* which visual structures support the chosen state;
* which competing rows were considered;
* whether any competing row implies a conflicting action;
* whether the decisive evidence is present;
* whether the observation remains within the declared operating domain;
* whether the decision should be accepted or rejected.

R1 improves the first-stage address ranking but does not yet provide enough evidence for those governance questions.

This failure is informative rather than merely negative.

It indicates that the next system should produce multiple localized pieces of evidence rather than one aggregate similarity score.

---

## Next experiment

The next declared experiment is:

```text
translation_equivariant_template_correlation
```

Its purpose is not merely to increase top-1 accuracy.

It should test whether several spatially localized template matches can create a more useful separation between correct rows and dangerous near-matches.

The next provider should preserve:

* native-resolution operation;
* deterministic execution;
* explicit spatial coordinates;
* no final-evaluation tuning;
* exact-row and action metrics;
* distinguishable rejection controls;
* conflicting-action safety checks;
* complete trace reconstruction;
* frozen evidence identities.

It should add evidence such as:

```text
matched local regions
region coordinates
per-region correlation scores
expected versus observed region layout
missing critical regions
contradictory local regions
best competing-action evidence
spatial consistency
```

The central Stage 2 question should be:

> Can translation-equivariant local correlation produce a nontrivial accepted benign region while preserving zero observed distinguishable false accepts and zero accepted conflicting-action errors on the declared fixtures?

---

## Stage 2 success conditions

A useful Stage 2 result should require:

```text
zero observed distinguishable false accepts
zero accepted conflicting-action errors
nonzero transferred benign coverage
```

The existing decision ladder remains appropriate:

### Outcome A

```text
final benign accepted coverage >= 50%
```

Proceed to broader robustness validation.

### Outcome B

```text
10% <= final benign accepted coverage < 50%
```

Local evidence helps but remains incomplete.

Proceed to deterministic geometry extraction while preserving the useful local evidence.

### Outcome C

```text
final benign accepted coverage < 10%
or no feasible transferred operating point
```

Proceed to deterministic connected components and explicit geometry.

### Invalid

Use invalid when:

* calibration leakage occurs;
* required controls are missing;
* frozen identities do not verify;
* evidence cannot be reconstructed;
* final evaluation precedes calibration freeze;
* required artifacts are incomplete.

---

## Research sequence from here

The current evidence supports the following order:

```text
Phase A
Global visual-address baselines
Result:
strong raw action ranking, weak exact-row ranking,
no useful governed operating point

Phase B Stage 1
Bounded registration plus local normalized pixels
Result:
translation failure substantially repaired,
exact-row ranking improved,
governed acceptance still collapsed

Phase B Stage 2
Translation-equivariant local template correlation
Question:
can multiple local matches create discriminative evidence?

Phase B Stage 3
Deterministic connected-components and geometry extraction
Question:
can explicit objects, positions and relationships separate
states that remain ambiguous at the pixel-correlation level?

Only after deterministic stages fail:
tiny native-resolution learned representation

Only after the learned local baseline is measured:
factorized Visual State Compiler
```

The current evidence does not justify jumping directly to a large learned visual model.

It does justify continuing the local, factorized direction.

---

## Strongest honest claim

The strongest current claim is:

> On the declared bounded synthetic arcade fixture, deterministic integer registration increased raw exact policy-row accuracy from `75%` to `87.5%`, increased raw action accuracy from `96.875%` to `98.4375%`, and completely repaired the held-out two-pixel translation family at top-1. The current registered-pixel calibration did not produce a useful governed operating point: it preserved zero observed distinguishable false accepts and zero accepted conflicting-action errors but accepted none of the `1,344` benign final observations.

---

## Overall conclusion

ZeroModel has not yet proved a useful end-to-end visual-AI reader.

It has proved several components needed for one:

```text
the pixels contain policy-state information;
local alignment recovers information hidden by global comparison;
bounded translation can be handled deterministically;
action recognition is already very strong;
exact-row recognition is improving;
evaluation and evidence lineage are auditable.
```

The remaining bottleneck is not basic visual guessing.

It is trustworthy evidence.

The next phase must move from:

```text
this whole image resembles row X
```

toward:

```text
these localized structures were found;
they occupy the expected positions;
their relationships support row X;
the critical evidence for row X is present;
the strongest competing-action evidence is insufficient;
therefore the visual address may be accepted.
```

That transition—from similarity to structured evidence—is the next major test of the ZeroModel visual architecture.

---

## Related files

Experiment design and result:

```text
docs/research/visual-local-baseline-showdown.md
```

Frozen Stage 1 evidence:

```text
docs/results/visual-local-baseline-showdown-v1/
```

Frozen System B comparator:

```text
docs/results/visual-address-system-b-v2/
```

Claims ledger:

```text
docs/claims-audit.md
```
