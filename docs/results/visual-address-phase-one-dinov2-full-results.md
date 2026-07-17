# Phase 1 Full Pinned DINOv2 Visual-Address Benchmark Results

**Document status:** Full measured research result  
**Repository:** `ernanhughes/zeromodel`  
**Date:** 17 July 2026  
**Validation status:** `research`  
**Dataset digest:** `91b1b422482eeeef20eb182162eb2a745f9b50524cc7f94ec95a0aba5f2fa37e`  
**Report digest:** `d7d0b4db13c9f96b2ac4583aae25fa2159fb62471fb90110103548317f084035`  
**Recommended repository path:** `docs/results/visual-address-phase-one-dinov2-full.md`

---

## 1. Executive decision

This document records the first full pinned DINOv2 run of ZeroModel's Phase 1
family-held-out visual-address benchmark.

The benchmark produced a decisive negative result for the current learned
visual-address direction:

> **Frozen DINOv2 representations did not materially improve benign policy-action
> recovery, substantially degraded exact policy-row addressing, and produced
> unsafe false-acceptance rates.**

System D, the strongest learned system by benign action accuracy, recovered the
correct action on `74.78%` of benign observations. The simple normalized-pixel
baseline B recovered `73.44%`, a difference of only `1.34` percentage points.
That small action-level difference came with materially worse properties:

- exact benign row recovery fell from `62.50%` for B to `30.36%` for D;
- false acceptance rose from `51.21%` to `73.79%`;
- accepted benign action correctness fell from `97.92%` to `77.49%`;
- conflicting-action errors rose from `21` to `292`;
- D retained every prototype representation rather than one medoid per row.

The experiment was designed to evaluate **observation-addressed policy**, not
merely four-class action prediction. On that intended task, the normalized-pixel
baseline is clearly stronger than every frozen-embedding system tested.

The result therefore does **not** justify promotion to policy-aware projection,
critical-mask verification, local patch verification, NetVLAD, physical
deployment, or a broader Level 2 visual architecture.

The provider-neutral address seam remains useful. The tested learned
implementations should remain research baselines behind that seam rather than
becoming the project's promoted runtime architecture.

---

## 2. Research question

The Phase 1 benchmark asked two separate questions:

1. Can a visual representation recover the correct policy row and action under
   held-out benign visual variation?
2. Can the system reject distinguishable missing-evidence and out-of-domain
   observations rather than confidently inventing a valid policy address?

These questions must remain separate.

A system may improve apparent coverage by accepting more observations, while
simultaneously becoming less safe and less faithful to the governed policy
address. Likewise, a system may select the correct action while resolving the
wrong policy row because many of the 112 rows share one of only four actions.

The benchmark therefore evaluates:

- exact policy-row recovery;
- policy-action recovery;
- conflicting-action errors;
- false acceptance over distinguishable rejection opportunities;
- false rejection over benign opportunities;
- rejection and action behaviour separately;
- information-theoretic controls separately from scored rejection cases.

---

## 3. Systems evaluated

| ID | System | Representation and matcher |
|---|---|---|
| **A** | Current deterministic reader | Existing exact feature-codeword reader with first-class rejection |
| **B** | Normalized-pixel medoids | Mean-centred, L2-normalized grayscale pixels with one cosine medoid per policy row |
| **C** | Frozen embedding medoids | Pinned DINOv2-small CLS representation with one cosine medoid per policy row |
| **D** | Raw frozen-embedding k-NN | Pinned DINOv2-small representation retaining every prototype vector |
| **G** | Rejection-equipped linear probe | Ridge least-squares row classifier over the same pinned DINOv2 representations |

The learned path uses:

- model: `facebook/dinov2-small`;
- pinned revision: `ed25f3a31f01632728cabb09d1542f84ab7b0056`;
- representation: L2-normalized CLS token from `last_hidden_state`;
- crop-safe square letterboxing for the wide arcade frame;
- an identity-bearing encoder manifest;
- independent prototype and calibration splits;
- per-row score thresholds;
- per-row conflicting-action margins.

Systems C, D, and G reuse one frozen representation extraction. Differences
between them therefore arise from the addressing and fitting strategy rather
than from separate encoder runs.

---

## 4. Dataset composition

The full run used `variants_per_family = 3`.

| Partition | Observations | Scored in headline evaluation? |
|---|---:|---|
| Prototype families | 1,344 | No |
| Calibration families | 1,344 | No |
| Held-out benign test families | 1,344 | Yes: expected accept |
| Target-removal information-theoretic controls | 98 | No accept/reject expectation |
| Distinguishable critical interventions | 224 | Yes: expected reject |
| Out-of-domain observations | 24 | Yes: expected reject |
| **Total observations** | **4,378** | |
| **Total scored evaluation observations** | **1,592** | |

### 4.1 Prototype families

- clean intensity jitter;
- lower brightness;
- one-pixel vertical translation;
- palette A.

### 4.2 Calibration families

- held-out contrast;
- opposite vertical translation;
- palette B;
- low-amplitude integer noise.

### 4.3 Benign test families

- stronger unseen brightness;
- two-pixel vertical translation;
- palette C;
- a structured patch placed only over source-background pixels.

### 4.4 Distinguishable critical interventions

- tank removed;
- cooldown indicator removed.

These frames are visibly outside the complete valid-state renderer and are
scored as rejection opportunities.

### 4.5 Information-theoretic control

The target-removal family is not included in false-acceptance or
false-rejection denominators.

Removing a visible target can produce pixels identical to a valid no-target
state. No single-frame reader can distinguish hidden target presence from true
target absence when the pixel observations are identical. Temporal, multiview,
or independent sensor evidence would be required.

### 4.6 Out-of-domain families

- blank frame;
- checkerboard;
- impossible frame containing two spatially separated tanks.

---

## 5. Run identity

```text
dataset_digest:
91b1b422482eeeef20eb182162eb2a745f9b50524cc7f94ec95a0aba5f2fa37e

report_digest:
d7d0b4db13c9f96b2ac4583aae25fa2159fb62471fb90110103548317f084035

observation_count:
4,378

scored_evaluation_count:
1,592

validation_status:
research
```

The digest pair identifies the dataset manifest and benchmark report associated
with the numbers in this document.

The raw generated JSON report should be committed beside this document. This
Markdown file is an adjudication and interpretation record; it is not a
replacement for the machine-readable result.

---

## 6. Metric semantics

The console's existing `action_accuracy` and `row_accuracy` divide by all
`1,592` scored evaluation observations. That denominator includes both:

- `1,344` benign observations expected to be accepted; and
- `248` distinguishable observations expected to be rejected.

Those whole-evaluation rates are preserved in the raw report, but they are not
the clearest measurements of benign policy recovery.

This document derives the following explicit metrics.

### 6.1 Benign action accuracy

```text
correct_action_count / 1,344 benign opportunities
```

This measures whether the correct policy action was recovered on held-out
benign observations, whether or not the exact row was recovered.

### 6.2 Benign exact-row accuracy

```text
correct_row_count / 1,344 benign opportunities
```

This is the primary governed-address measurement. It asks whether the system
recovered the exact policy row associated with the observation.

### 6.3 False-acceptance rate

```text
false_accept_count / 248 distinguishable rejection opportunities
```

The 98 target-removal information-theoretic controls are intentionally excluded.

### 6.4 False-rejection rate

```text
false_reject_count / 1,344 benign opportunities
```

### 6.5 Accepted benign action correctness

```text
correct_action_count / accepted benign observations
```

This measures how trustworthy the system was when it chose to accept a benign
observation.

### 6.6 Action-equivalent wrong-row count

```text
correct_action_count - correct_row_count
```

This counts benign observations where the selected policy action was correct
but the exact policy address was not. It exposes action aliasing across rows.

### 6.7 End-to-end scored success

For diagnostic comparison only, this document also reports:

```text
(correct benign actions + correct distinguishable rejections) / 1,592
```

This is a post-run derived diagnostic, not a replacement for the predeclared
separate FAR, FRR, row, and action measurements.

---

## 7. Primary results

| System | Description | Benign action | Exact benign row | FAR | FRR | Conflicting-action errors |
|---|---|---:|---:|---:|---:|---:|
| **A** | current deterministic reader | 0.00% | 0.00% | 0.00% | 100.00% | 0 |
| **B** | normalized-pixel medoid/template matching | 73.44% | 62.50% | 51.21% | 25.00% | 21 |
| **C** | DINOv2 medoid retrieval | 70.01% | 30.73% | 82.26% | 3.27% | 359 |
| **D** | DINOv2 all-prototype k-NN | 74.78% | 30.36% | 73.79% | 3.50% | 292 |
| **G** | rejection-equipped ridge linear probe | 59.45% | 28.79% | 100.00% | 1.79% | 521 |

### 7.1 Results with 95% Wilson intervals

| System | Benign action accuracy | Exact benign row accuracy | False-acceptance rate | False-rejection rate |
|---|---:|---:|---:|---:|
| **A** | 0.00% (0.0–0.3%) | 0.00% (0.0–0.3%) | 0.00% (0.0–1.5%) | 100.00% (99.7–100.0%) |
| **B** | 73.44% (71.0–75.7%) | 62.50% (59.9–65.0%) | 51.21% (45.0–57.4%) | 25.00% (22.8–27.4%) |
| **C** | 70.01% (67.5–72.4%) | 30.73% (28.3–33.2%) | 82.26% (77.0–86.5%) | 3.27% (2.4–4.4%) |
| **D** | 74.78% (72.4–77.0%) | 30.36% (28.0–32.9%) | 73.79% (68.0–78.9%) | 3.50% (2.6–4.6%) |
| **G** | 59.45% (56.8–62.0%) | 28.79% (26.4–31.3%) | 100.00% (98.5–100.0%) | 1.79% (1.2–2.6%) |

Wilson intervals are included because FAR and FRR are binomial rates with
finite denominators. They are descriptive intervals for each system. The same
observations were evaluated by all systems, so a proper pairwise significance
test requires paired per-example outcomes rather than treating the systems as
independent samples.

---

## 8. Acceptance quality

| System | Accepted benign | Correct action when accepting | Exact row when accepting | Correct action from wrong row | Wrong accepted actions |
|---|---:|---:|---:|---:|---:|
| **A** | 0 | — | — | 0 | 0 |
| **B** | 1,008 | 97.92% | 83.33% | 147 | 21 |
| **C** | 1,300 | 72.38% | 31.77% | 528 | 359 |
| **D** | 1,297 | 77.49% | 31.46% | 597 | 292 |
| **G** | 1,320 | 60.53% | 29.32% | 412 | 521 |

This table reveals the central behavioural difference.

### System B

B rejects one quarter of benign observations, but when it accepts:

- `97.92%` of accepted benign observations receive the correct action;
- `83.33%` resolve to the exact policy row;
- only `21` accepted benign observations produce a conflicting action.

### Learned systems

C, D, and G achieve low false-rejection rates mainly by accepting almost
everything.

Their accepted decisions are much less reliable:

- C: `72.38%` correct action when accepting;
- D: `77.49%`;
- G: `60.53%`.

Low FRR is not evidence of robustness when it is purchased by unsafe acceptance.

---

## 9. The key result: action prediction is not policy addressing

The bounded arcade policy contains:

- 112 distinct policy rows;
- only four possible actions.

Many incorrect rows therefore share the same action.

The full run produced the following action-equivalent wrong-row counts:

| System | Correct actions | Exact rows | Correct action from wrong row |
|---|---:|---:|---:|
| A | 0 | 0 | 0 |
| B | 987 | 840 | 147 |
| C | 941 | 413 | 528 |
| D | 1,005 | 408 | 597 |
| G | 799 | 387 | 412 |

System D appears strongest if the task is reduced to four-class action
prediction. It appears weak when evaluated against the actual Phase 1 contract:

> recover the exact governed policy address, or reject.

D's benign action accuracy is `74.78%`, but its exact row accuracy is only
`30.36%`. More than half of its correct actions—`597` of `1,005`—come from the
wrong policy row.

This matters because ZeroModel's proposed benefit is not merely action
selection. The address is expected to preserve:

- exact artifact-row identity;
- row-specific evidence;
- replay;
- alternative actions;
- calibration provenance;
- policy-version-specific traceability.

A wrong row that happens to share the correct action does not provide that
contract.

The project should not redefine success after observing the result by allowing
unrecorded action-equivalent aliases. A many-observations-to-one-action system
would be a different, weaker architecture requiring a new benchmark and claim.

---

## 10. Safety result

All approximate systems failed the rejection requirement at their current
calibrated operating point.

| System | False accepts | Opportunities | FAR | Correct rejections |
|---|---:|---:|---:|---:|
| A | 0 | 248 | 0.00% | 248 |
| B | 127 | 248 | 51.21% | 121 |
| C | 204 | 248 | 82.26% | 44 |
| D | 183 | 248 | 73.79% | 65 |
| G | 248 | 248 | 100.00% | 0 |

Even the strongest simple baseline accepted more than half of the
distinguishable critical-intervention and OOD cases.

The frozen systems were worse:

- C accepted more than four out of five rejection opportunities;
- D accepted nearly three out of four;
- G accepted every rejection opportunity.

No system B/C/D/G is suitable for deployment under this operating point.

The deterministic reader A correctly refuses every distinguishable rejection
case, but also refuses every benign perturbation. Its result should be
interpreted as exact-codeword safety, not perturbation tolerance.

---

## 11. Diagnostic combined outcomes

| System | Correct distinguishable rejections | Correct disposition rate | End-to-end scored success |
|---|---:|---:|---:|
| **A** | 248 / 248 | 15.58% | 15.58% |
| **B** | 121 / 248 | 70.92% | 69.60% |
| **C** | 44 / 248 | 84.42% | 61.87% |
| **D** | 65 / 248 | 85.55% | 67.21% |
| **G** | 0 / 248 | 82.91% | 50.19% |

`Correct disposition rate` counts benign acceptance as disposition-correct even
when the accepted action is wrong. For this reason, it can make systems that
accept broadly appear stronger than they are.

`End-to-end scored success` requires both:

- a correct action on benign observations; or
- a correct rejection on distinguishable rejection observations.

On that derived diagnostic, B is strongest at `69.60%`, followed by D at
`67.21%`. This comparison still does not repair B's `51.21%` FAR or D's
`73.79%` FAR. Separate safety and fidelity metrics remain the governing result.

---

## 12. System-by-system adjudication

### 12.1 System A — current deterministic reader

**Result**

- benign action accuracy: `0.00%`;
- benign exact row accuracy: `0.00%`;
- FAR: `0.00%`;
- FRR: `100.00%`.

**Interpretation**

The exact deterministic reader has no tolerance for the tested held-out
families. It preserves strict refusal rather than guessing.

This does not invalidate the previously validated canonical result. It sharpens
the boundary:

- canonical exact feature-codeword addressing remains validated;
- ordinary perturbation tolerance is refuted for these tested families;
- A should remain available where exact canonical observations are guaranteed.

### 12.2 System B — normalized-pixel medoids

**Result**

- benign action accuracy: `73.44%`;
- benign exact row accuracy: `62.50%`;
- FAR: `51.21%`;
- FRR: `25.00%`;
- accepted-action correctness: `97.92%`;
- conflicting-action errors: `21`.

**Interpretation**

B is the strongest overall addressing baseline in this experiment.

It does not have the highest point estimate for benign action accuracy, but it:

- doubles the exact-row accuracy of C, D, and G;
- produces dramatically fewer conflicting-action errors;
- is far more reliable when accepting;
- has a substantially lower FAR than every learned system.

B is nevertheless unsafe. A `51.21%` FAR rules out deployment and rules out a
claim of safe approximate addressing.

### 12.3 System C — DINOv2 medoids

**Result**

- benign action accuracy: `70.01%`;
- benign exact row accuracy: `30.73%`;
- FAR: `82.26%`;
- FRR: `3.27%`;
- accepted-action correctness: `72.38%`;
- conflicting-action errors: `359`.

**Interpretation**

C wins the narrow comparison against G, but does not survive the broader Phase
1 decision.

Relative to B, C:

- loses `3.42` percentage points of benign action accuracy;
- loses `31.77` points of exact-row accuracy;
- increases FAR by `31.05` points;
- produces `338` additional conflicting-action errors.

The low FRR reflects broad acceptance, not trustworthy generalization.

### 12.4 System D — DINOv2 all-prototype k-NN

**Result**

- benign action accuracy: `74.78%`;
- benign exact row accuracy: `30.36%`;
- FAR: `73.79%`;
- FRR: `3.50%`;
- accepted-action correctness: `77.49%`;
- conflicting-action errors: `292`.

**Interpretation**

D is the strongest learned system on benign action accuracy and exceeds B by
`1.34` percentage points.

That gain is not sufficient to justify the trade:

- exact-row accuracy is `32.14` points lower than B;
- FAR is `22.58` points higher;
- accepted-action correctness is `20.43` points lower;
- conflicting-action errors increase from `21` to `292`;
- D retains all prototype vectors rather than one medoid per row.

The 95% Wilson intervals for B and D benign action accuracy overlap. Because the
same observations were used, the correct next statistical comparison would be a
paired test over per-example outcomes. Even a statistically reliable action
difference would not erase the much larger address and safety regressions.

D does not establish that frozen embeddings materially improve governed visual
addressing.

### 12.5 System G — rejection-equipped linear probe

**Result**

- benign action accuracy: `59.45%`;
- benign exact row accuracy: `28.79%`;
- FAR: `100.00%`;
- FRR: `1.79%`;
- accepted-action correctness: `60.53%`;
- conflicting-action errors: `521`.

**Interpretation**

G is decisively eliminated.

It combines:

- the weakest benign action result among approximate systems;
- the weakest exact-row recovery;
- acceptance of every distinguishable rejection opportunity;
- the largest number of conflicting-action errors.

The linear probe does not provide an accuracy-equivalent, lower-complexity
replacement for retrieval.

---

## 13. Kill-condition adjudication

The Phase 1 protocol declared explicit reasons to reduce or stop the direction.

| Kill condition | Status after full run | Evidence |
|---|---|---|
| Simpler normalized pixels or deterministic reader match frozen embeddings in the operating region | **Triggered** | B is within 1.34 action points of D, while beating every frozen system substantially on exact-row recovery, FAR, and accepted-decision reliability |
| Linear probe matches retrieval with materially less complexity | **Failed for G / G eliminated** | G has 59.45% benign action accuracy, 28.79% row accuracy, 100% FAR, and 521 conflicting errors |
| Distinguishable critical interventions are accepted at an unsafe rate | **Triggered** | FAR ranges from 51.21% to 100% for B/C/D/G |
| Held-out family calibration transfers adequately | **Not supported** | Low exact-row recovery and high FAR show that current calibration does not transfer safely; raw per-family tables should be retained for the final audit |
| Governance wrapper reaches parity at lower complexity | **Not evaluated** | Accuracy and safety already block promotion; governance parity is no longer required before stopping the current path |
| Prototype count approaches one stored vector per observation without compensating benefit | **Strong pressure to trigger for D** | D retains all prototypes for a 1.34-point action gain over B while losing 32.14 row points and 22.58 FAR points |

### Phase decision

The current Phase 1 learned path does not pass continuation criteria.

---

## 14. Claim-status transitions

| Claim | Previous status | Status after full run |
|---|---|---|
| Canonical closed-world exact-codeword addressing works | Validated | **Remains validated** |
| The deterministic reader tolerates ordinary benign perturbations | Not established | **Refuted for the tested families** |
| Normalized pixels provide safe approximate addressing | Research hypothesis | **Refuted at the tested operating point** |
| DINOv2 medoids improve policy addressing | Research hypothesis | **Not supported** |
| DINOv2 all-prototype k-NN materially beats simple pixels | Research hypothesis | **Not supported** |
| A rejection-equipped linear probe matches retrieval at lower complexity | Research hypothesis | **Refuted for this fixture** |
| Tested approximate systems safely reject missing evidence and OOD frames | Research hypothesis | **Refuted at the tested operating point** |
| Learned visual addressing is deployment-ready | Not claimed | **Explicitly not supported** |
| A provider-neutral visual-address seam is useful | Implemented architecture | **Remains valid** |
| Frozen-encoder systems can be implemented behind the seam | Implemented | **Measured as research baselines, not promoted** |

The claims audit should distinguish **evidence against** from mere absence of
evidence. A result that refutes a tested claim should not remain labelled only
“not yet validated.”

---

## 15. Architectural decision

### Retain

- `ImageObservation`;
- `VisualAddressContract`;
- `VisualAddressDecision`;
- `VisualAddressProvider`;
- separate policy and address identities;
- `MatrixBlob`;
- encoder and preprocessing manifests;
- independent calibration artifacts;
- first-class rejection;
- family-held-out benchmark machinery;
- exact deterministic canonical reader;
- systems B/C/D/G as conventional research baselines.

### Do not promote

- DINOv2 as the preferred visual-address representation;
- medoid retrieval as a deployment path;
- all-prototype k-NN as a deployment path;
- the linear probe;
- a learned Level 2 visual runtime;
- critical-mask or patch verification as the automatic next stage;
- a claim of learned visual generalization;
- physical control or bridge deployment.

### Deferred research

A future experiment may revisit visual addressing only with a materially new
hypothesis, such as:

- a representation trained specifically for exact policy-row identity;
- explicit local evidence-presence verification;
- temporal evidence for information-theoretically ambiguous frames;
- a different bounded environment where global semantic embeddings match the
  task;
- calibrated abstention optimized against a declared safety loss.

Such work should begin as a new research branch, not as an assumed continuation
of the current architecture.

---

## 16. Required repository changes

The next PR should be an **evidence and adjudication PR**, not a representation
expansion PR.

### 16.1 Commit the raw evidence

Recommended files:

```text
docs/results/visual-address-phase-one-dinov2-full.md
docs/results/visual-address-phase-one-dinov2-full.json
```

The JSON file should be the original generated report, not only the console
summary.

### 16.2 Add confidence intervals to report serialization

`VisualBenchmarkMetrics.to_dict()` should include Wilson intervals for:

- false-acceptance rate;
- false-rejection rate;
- benign action accuracy;
- benign exact-row accuracy.

Counts and denominators must remain primary.

### 16.3 Promote benign-denominator metrics

The report currently exposes whole-evaluation `action_accuracy` and
`row_accuracy`. Add explicit first-class fields for:

- `benign_action_accuracy`;
- `benign_row_accuracy`;
- `accepted_benign_action_correctness`;
- their numerators and denominators.

### 16.4 Preserve per-family outcomes

The full report should retain raw counts for every family. The console extract
used to write this document does not expose those tables, so this document does
not invent them.

Before merge, verify and preserve:

- acceptance and rejection counts per family;
- exact-row and action recovery per benign family;
- critical-tank FAR;
- critical-cooldown FAR;
- blank/checkerboard/impossible-state FAR;
- target-removal control acceptance and rejection counts.

### 16.5 Fix the no-conflicting-action contract mismatch

When a policy region has no conflicting-action candidate:

- calibration should not synthesize a margin that runtime cannot reproduce;
- runtime should skip the conflicting-action margin check;
- the decision trace should explicitly record that no conflicting candidate
  exists;
- `second_row_id`, `second_score`, and ambiguity measure should be optional.

### 16.6 Scope representation digests

The current frozen-vector digest should be documented as identifying:

> the exact float32 representation produced in the recorded execution
> environment.

It should not be described as reproducible across devices, BLAS libraries,
PyTorch versions, or CPU/GPU backends.

An int8 quantization ADR remains appropriate before making cross-machine vector
identity claims.

### 16.7 Update the claims audit

Record the claim transitions in Section 14 and state clearly that the full
Phase 1 learned path failed continuation criteria.

---

## 17. Reproducibility manifest

The result should not be merged as a cross-machine evidence record until the
local execution environment is captured.

Fill every required field below from the machine that produced the report.

| Field | Recorded value |
|---|---|
| Git commit SHA | **[REQUIRED]** |
| Git branch or detached ref | **[REQUIRED]** |
| Exact command | **[REQUIRED]** |
| Operating system and version | **[REQUIRED]** |
| Machine architecture | **[REQUIRED]** |
| CPU | **[REQUIRED]** |
| GPU, accelerator, or CPU-only | **[REQUIRED]** |
| RAM | **[REQUIRED]** |
| Python version | **[REQUIRED]** |
| NumPy version | **[REQUIRED]** |
| PyTorch version | **[REQUIRED]** |
| Torchvision version | **[REQUIRED]** |
| Transformers version | **[REQUIRED]** |
| Hugging Face Hub version | **[REQUIRED]** |
| ZeroModel install mode/version | **[REQUIRED]** |
| DINOv2 model ID | `facebook/dinov2-small` |
| DINOv2 revision | `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| Encoder manifest ID | **[COPY FROM RAW REPORT]** |
| Loaded weights digest | **[COPY FROM RAW REPORT]** |
| Preprocessing digest | **[COPY FROM RAW REPORT]** |
| Dataset digest | `91b1b422482eeeef20eb182162eb2a745f9b50524cc7f94ec95a0aba5f2fa37e` |
| Report digest | `d7d0b4db13c9f96b2ac4583aae25fa2159fb62471fb90110103548317f084035` |
| Model cache/download state | **[REQUIRED]** |
| `variants_per_family` | `3` |
| Random seeds | **[COPY FROM MANIFESTS]** |

Canonical command, if it matches the actual local invocation:

```bash
python examples/arcade_visual_address_benchmark.py \
  --encoder dinov2 \
  --output-dir build/visual-phase-one
```

Replace it with the exact command actually used.

---

## 18. Limitations

### 18.1 Bounded synthetic fixture

The benchmark uses a small, enumerable arcade renderer. It is not evidence about
general photographs, cameras, natural images, or open-world perception.

### 18.2 One encoder family

Only the pinned DINOv2-small representation was tested. The result does not
prove that every possible learned representation will fail.

It does prove that this tested global frozen representation does not justify
promotion.

### 18.3 One calibration strategy

The current run uses the declared per-row lower empirical quantile and
conflicting-action margin calibration. Alternative operating curves may produce
different FAR/FRR trade-offs.

No threshold choice can be described as a success without preserving exact-row
fidelity and bringing FAR into a declared safe range.

### 18.4 No paired significance test in this document

The console summary contains aggregate counts, not the paired per-observation
outcomes required for McNemar or other paired comparisons.

The B-versus-D action difference should therefore be described as a point
estimate, not a proven statistically significant win.

### 18.5 Per-family tables not reproduced here

The console output does not contain full per-family counts or the
information-theoretic-control disposition. Those should be read from and
committed with the original full JSON report.

### 18.6 Research status

The report remains `validation_status = "research"`. No deployment binding
should be promoted from it.

---

## 19. Final conclusion

The instrument worked.

It separated three questions that would otherwise have been conflated:

1. Does the system accept benign variation?
2. Does it recover the correct action?
3. Does it recover the exact governed policy address while safely rejecting
   distinguishable invalid observations?

The frozen systems accepted benign variation more readily than the exact reader.
They did not provide a safe or faithful visual address.

The strongest learned system, D:

- improved benign action accuracy over B by only `1.34` points;
- recovered the exact row `32.14` points less often;
- increased FAR by `22.58` points;
- produced `271` more conflicting-action errors;
- required an all-prototype index.

The result therefore supports a narrow and credible ZeroModel direction:

> **Compile and govern bounded policy artifacts; preserve an interchangeable
> address-provider seam; use exact deterministic observation addressing when the
> observation contract is exact; and require new evidence before promoting any
> learned visual-address mechanism.**

The benchmark should be regarded as a successful research outcome even though
the tested learned architecture failed. The project built the harness first,
published the unfavourable numbers, and allowed the numbers to stop an
attractive direction.

---

## Appendix A — Raw console summary

```json
{
  "dataset_digest": "91b1b422482eeeef20eb182162eb2a745f9b50524cc7f94ec95a0aba5f2fa37e",
  "observation_count": 4378,
  "report_digest": "d7d0b4db13c9f96b2ac4583aae25fa2159fb62471fb90110103548317f084035",
  "systems": {
    "A": {
      "accepted_count": 0,
      "action_accuracy": 0.0,
      "conflicting_action_error_count": 0,
      "correct_action_count": 0,
      "correct_row_count": 0,
      "evaluation_count": 1592,
      "false_accept_count": 0,
      "false_accept_opportunities": 248,
      "false_acceptance_rate": 0.0,
      "false_reject_count": 1344,
      "false_reject_opportunities": 1344,
      "false_rejection_rate": 1.0,
      "rejected_count": 1592,
      "row_accuracy": 0.0
    },
    "B": {
      "accepted_count": 1135,
      "action_accuracy": 0.6199748743718593,
      "conflicting_action_error_count": 21,
      "correct_action_count": 987,
      "correct_row_count": 840,
      "evaluation_count": 1592,
      "false_accept_count": 127,
      "false_accept_opportunities": 248,
      "false_acceptance_rate": 0.5120967741935484,
      "false_reject_count": 336,
      "false_reject_opportunities": 1344,
      "false_rejection_rate": 0.25,
      "rejected_count": 457,
      "row_accuracy": 0.5276381909547738
    },
    "C": {
      "accepted_count": 1504,
      "action_accuracy": 0.5910804020100503,
      "conflicting_action_error_count": 359,
      "correct_action_count": 941,
      "correct_row_count": 413,
      "evaluation_count": 1592,
      "false_accept_count": 204,
      "false_accept_opportunities": 248,
      "false_acceptance_rate": 0.8225806451612904,
      "false_reject_count": 44,
      "false_reject_opportunities": 1344,
      "false_rejection_rate": 0.03273809523809524,
      "rejected_count": 88,
      "row_accuracy": 0.2594221105527638
    },
    "D": {
      "accepted_count": 1480,
      "action_accuracy": 0.6312814070351759,
      "conflicting_action_error_count": 292,
      "correct_action_count": 1005,
      "correct_row_count": 408,
      "evaluation_count": 1592,
      "false_accept_count": 183,
      "false_accept_opportunities": 248,
      "false_acceptance_rate": 0.7379032258064516,
      "false_reject_count": 47,
      "false_reject_opportunities": 1344,
      "false_rejection_rate": 0.034970238095238096,
      "rejected_count": 112,
      "row_accuracy": 0.2562814070351759
    },
    "G": {
      "accepted_count": 1568,
      "action_accuracy": 0.5018844221105527,
      "conflicting_action_error_count": 521,
      "correct_action_count": 799,
      "correct_row_count": 387,
      "evaluation_count": 1592,
      "false_accept_count": 248,
      "false_accept_opportunities": 248,
      "false_acceptance_rate": 1.0,
      "false_reject_count": 24,
      "false_reject_opportunities": 1344,
      "false_rejection_rate": 0.017857142857142856,
      "rejected_count": 24,
      "row_accuracy": 0.24309045226130654
    }
  },
  "validation_status": "research"
}
```

---

## Appendix B — Derived metric formulas

```text
benign_action_accuracy
    = correct_action_count / false_reject_opportunities

benign_exact_row_accuracy
    = correct_row_count / false_reject_opportunities

accepted_benign_count
    = false_reject_opportunities - false_reject_count

accepted_benign_action_correctness
    = correct_action_count / accepted_benign_count

accepted_benign_exact_row_correctness
    = correct_row_count / accepted_benign_count

action_equivalent_wrong_row_count
    = correct_action_count - correct_row_count

true_reject_count
    = false_accept_opportunities - false_accept_count

end_to_end_scored_success
    = (correct_action_count + true_reject_count) / evaluation_count
```

Wilson score intervals use `z = 1.959963984540054` for nominal 95% coverage.

---

## Appendix C — Related repository records

- `docs/research/visual-address-phase-one.md`
- `docs/claims-audit.md`
- `examples/arcade_visual_address_benchmark.py`
- `.github/scripts/run_visual_address_smoke.py`
- `.github/workflows/visual-address-benchmark.yml`
- `zeromodel/visual_retrieval.py`
- `zeromodel/visual_experiment.py`
- `zeromodel/visual_encoder.py`
