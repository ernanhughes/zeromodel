# Adversarial External Review — ZeroModel Episode-Family Semantics Repair

Please perform a hostile, implementation-level review of the current ZeroModel research branch.

This is not a request for general encouragement, documentation feedback, stylistic comments, or a summary of the pull request description.

Your task is to independently determine whether the branch actually repairs the scientific semantics of the prospective visual action-set benchmark—or whether it has replaced the previous defects with new success-shaped but vacuous checks.

---

# Repository and branch

Repository:

`https://github.com/ernanhughes/zeromodel`

Review branch:

`research/video-action-set-family-semantics-v2`

Base branch:

`main`

Expected base commit:

`6014c01554116bc8f538627fb49bd77168783d6e`

Expected branch commits:

```text
df09b9c stage3-reference: repair frame-invalid family semantics
b44153d stage3-reference: repair information-control ambiguity
677b209 stage3-reference: extend family verification and mutation audit
```

First confirm that the remote branch exists and that the commit sequence matches.

Use a clean clone or a fresh worktree.

Example:

```bash
git clone https://github.com/ernanhughes/zeromodel.git
cd zeromodel

git fetch origin
git switch --detach origin/research/video-action-set-family-semantics-v2

git log --oneline --decorate -5
git status --short
git diff --stat origin/main...HEAD
```

Do not review a stale local checkout or infer implementation from the PR title.

---

# Scientific context

ZeroModel is investigating whether visual observations can address a finite compiled policy while preserving:

* deterministic identity;
* complete evidence;
* explicit numerical ties;
* action-level ambiguity;
* rejection;
* temporal reachability;
* reproducible verification.

The benchmark uses a finite arcade policy universe containing 112 rows.

Each policy row represents a combination of:

```text
tank position
target position
cooldown state
```

Each row maps to one of:

```text
LEFT
RIGHT
STAY
FIRE
```

The project’s broader scientific question is whether a visual provider can preserve enough evidence to produce:

```text
visual observation
    ↓
plausible policy rows
    ↓
action-unanimous candidate set
    ↓
action or explicit rejection
```

This branch does not attempt to answer the full question.

It repairs only the semantics of the benchmark’s episode families.

---

# Why this branch exists

The previous merged repair, PR #49, substantially improved:

* content identity;
* causal seed derivation;
* concrete sealed episode plans;
* complete score-vector evidence;
* numerical tie handling;
* semantic top-set outcomes;
* independent evidence reconstruction;
* an executable adversarial mutation audit.

However, external review found a serious semantic failure.

The previous `conflicting_action_splice` replaced the target region from one valid state with the target region from another.

The resulting image was:

```text
primary tank
secondary target
primary cooldown
```

Because the 112-state policy universe contains every valid tank–target–cooldown combination, that supposedly invalid image could be a third canonical valid state.

The previous contract proved that:

* the output differed from source A;
* the output differed from source B;
* both sources contributed pixels;
* the source actions conflicted.

It failed to prove:

> The output was outside the complete valid observation universe.

External review also found that the earlier information-theoretic controls could contain only one hidden history or hidden label repeated several times.

That proves byte repetition, but not ambiguity.

This branch claims to repair those defects.

---

# Claimed scope of the branch

The branch claims to repair:

1. conflicting-action splice semantics;
2. canonical valid-state collision closure;
3. information-theoretic control ambiguity;
4. frame-level versus episode-level disposition;
5. final-observation provenance;
6. family-level independent verification;
7. family-level executable mutation coverage.

It must not claim to repair:

* action-conditioned reachability;
* actual-executed-action temporal composition;
* final-access enforcement at every lower-level entry point;
* provider formula conformance;
* P3 development identity;
* optimized provider execution;
* materialization authorization;
* calibration;
* architecture selection;
* candidate tuning;
* provider selection;
* benchmark utility;
* final evaluation.

The claimed status after this branch is:

```text
episode_family_semantics_correct
```

The broader statuses must remain:

```text
reference_instrument_correctness_unresolved
prospective_materialization_prohibited
```

---

# Reported implementation result

The author reports three commits:

```text
df09b9c
stage3-reference: repair frame-invalid family semantics

b44153d
stage3-reference: repair information-control ambiguity

677b209
stage3-reference: extend family verification and mutation audit
```

Reported tests:

```text
Focused family/instrument set:
28 passed

Non-slow reference verification:
14 passed, 1 deselected

Complete slow mutation audit:
1 passed, 14 deselected

Combined non-slow domain set:
42 passed, 1 deselected

Default repository suite:
372 passed, 27 skipped
```

Do not accept these results as evidence merely because they are reported.

Run the relevant tests and inspect what they actually prove.

---

# Primary review question

Determine whether the branch genuinely supports:

```text
episode_family_semantics_correct
```

The standard is not:

> The tests pass.

The standard is:

> Every repaired benchmark family now corresponds to the independently defined scientific phenomenon named by that family, and the verifier can detect violations of the decisive semantic properties.

---

# Required review areas

## 1. Canonical observation universe

Inspect how the branch defines the canonical valid observation universe.

Determine whether it binds:

* the complete 112-row policy universe;
* exact canonical observation pixels;
* observation pixel digests;
* digest-to-row mappings;
* possible duplicate visual groups;
* policy identity;
* renderer identity;
* universe identity.

Questions:

1. Is the universe actually complete?
2. Is it regenerated from authoritative code or trusted from stored output?
3. Can a stale or altered universe be silently accepted?
4. If two valid rows have identical rendered pixels, are all associated rows preserved?
5. Is invalidity checked against the entire universe rather than against only source observations?
6. Is the comparison based on actual emitted pixels rather than descriptive metadata?

Try to alter one canonical observation or remove one row and determine whether verification detects it.

---

## 2. Repaired conflicting-action splice

Inspect the new splice implementation in detail.

Determine exactly what pixels are taken from:

* the primary source;
* the secondary source;
* any generated or synthetic component.

Establish whether the output contains simultaneously visible, action-relevant evidence from both source observations.

Questions:

1. Are source actions genuinely different?
2. Are source pairs selected deterministically?
3. Does each source contribute effective pixels?
4. Does each source contribute policy-relevant pixels?
5. Can one contribution be present only in a technically changed but visually irrelevant region?
6. Can the new composition equal either source?
7. Can it equal a third valid canonical observation?
8. Can it decode to a valid policy row despite having a new digest?
9. Can it equal a declared valid transformed observation?
10. Is the resulting contradiction visually meaningful, or is it arbitrary corruption labelled “conflicting action”?

Do not accept a check such as:

```python
output != primary and output != secondary
```

as proof of invalidity.

The decisive invariant is:

```text
output ∉ complete valid observation universe
```

Reproduce the branch’s collision measurement independently.

Materialize the relevant non-final fixtures into a temporary directory and compare every frame-invalid output against every canonical valid observation.

Report exact counts:

```text
generated frame-invalid outputs
canonical pixel collisions
valid-row decodes
source-A equality
source-B equality
third-valid-state collisions
non-no-op outputs
closure passes
closure failures
```

---

## 3. Global frame-invalid closure

Determine whether the branch introduces a general invariant for all families declared as distinguishable frame-invalid inputs.

At minimum inspect:

* `conflicting_action_splice`;
* `critical_evidence_corruption`.

Questions:

1. Is the collision rule general or hardcoded only for the splice family?
2. Would a future frame-invalid family automatically receive the same protection?
3. Is the closure enforced during generation, verification, or both?
4. Can an invalid record be created before the guard runs?
5. Can a caller bypass the closure through a lower-level function?
6. Are stored collision booleans independently reconstructed?
7. Are outer digests updated in mutation tests so semantic verification is required?

The closure should detect a deliberately laundered invalid frame whose pixels are replaced with a valid canonical prototype and whose surrounding digests are recomputed.

---

## 4. Correct distinction between frame invalidity and sequence invalidity

The benchmark also contains temporal-negative families such as:

* reordered frames;
* stale repeated frames;
* impossible transitions;
* declared gaps or unknown actions.

An individually valid frame may participate in an invalid sequence.

Determine whether the branch correctly separates:

```text
frame_disposition
episode_disposition
family
denominator class
```

Questions:

1. Are temporal negatives allowed to contain canonical valid frames?
2. Does the new non-collision rule incorrectly reject temporal-negative frames?
3. Can a temporal-negative episode be labelled valid because all individual frames are valid?
4. Can a frame-invalid episode be labelled only temporal-negative?
5. Does denominator selection use the intended family/episode classification?
6. Can a naive consumer accidentally count temporal-negative frames as valid evidence?

Report the exact disposition fields and validation rules.

---

## 5. Information-theoretic control semantics

Inspect the repaired information-control construction.

A valid information-theoretic control requires at least two genuinely distinct hidden causes or histories that map to the same complete observer-visible input.

The requirement is not satisfied by:

* repeating one observation;
* assigning two arbitrary labels to one history;
* changing only an ID the observer can see;
* declaring two sources distinct when their latent causal histories are identical.

Determine the exact observer model.

Questions:

1. What information is the tested provider allowed to observe?
2. Is the control about a single-frame observer, a sequence observer, or another explicit observer?
3. What constitutes a hidden history?
4. Are there at least two distinct predecessor/action histories?
5. Do those histories converge to exactly the same current visible input?
6. Are hidden labels derived from real causal differences?
7. Are all provider-visible pixels byte-identical?
8. Are all provider-visible semantic fields identical?
9. Can source IDs, frame IDs, filenames, ordering, metadata, or descriptors leak the hidden history?
10. Is denominator eligibility still false?
11. Does the verifier independently reconstruct hidden-history cardinality?
12. What happens if no grounded ambiguous control exists?

Independently report, per control group:

```text
member count
distinct hidden-history count
distinct hidden-label count
visible pixel digest count
provider-visible payload identity count
leakage indicators
denominator eligibility
```

A valid group should normally establish:

```text
hidden-history count >= 2
visible payload count == 1
```

Try mutating a control group so all members share one hidden history while maintaining surrounding digests.

The verifier should fail with a specific semantic failure code.

---

## 6. Hidden-information leakage

The branch may claim that hidden histories are invisible to the provider.

Verify this rather than trusting the claim.

Inspect the complete runtime/provider input surface.

Check whether the provider receives or could derive:

* frame ID;
* episode ID;
* source row;
* predecessor row;
* expected action;
* actual action;
* filename;
* ordering;
* source scope;
* transformation metadata;
* hidden-history digest;
* labels stored alongside pixels.

Determine whether the hidden distinction remains absent from everything the tested observer can access.

If different IDs are visible to the provider, byte-identical pixels alone do not establish indistinguishability.

---

## 7. Final-observation provenance

Inspect frames that undergo more than one operation, especially stale-repeat and temporal interventions.

Determine whether the recorded provenance chain can reconstruct the final emitted pixels.

Questions:

1. Is the original source digest recorded?
2. Is the transformed digest recorded?
3. Is the post-intervention digest recorded?
4. Is the final emitted digest recorded?
5. Are operations ordered?
6. Is the final operation explicit?
7. Can replay of the operation chain reconstruct the exact output?
8. Can a stale-repeat frame retain transformation metadata that no longer explains the emitted pixels?
9. Can the last intervention be omitted while outer digests are recomputed?
10. Does independent verification catch that omission?

Report whether:

```text
replayed final digest == emitted final observation digest
```

for every relevant record.

---

## 8. Independent verification quality

Determine whether the verifier reconstructs the semantic properties or trusts stored claims.

It must not simply trust fields such as:

```text
collides_with_valid = false
hidden_history_count = 2
control_ambiguity = true
provenance_valid = true
episode_disposition = temporal_negative
```

Questions:

1. Does it regenerate the canonical universe?
2. Does it recompute every frame-invalid collision?
3. Does it reconstruct source contributions?
4. Does it reconstruct hidden-history cardinality?
5. Does it reconstruct the provider-visible payload?
6. Does it replay final-observation provenance?
7. Does it derive disposition from family semantics?
8. Does it compare regenerated results against stored records?
9. Can self-consistent false metadata survive?
10. Can digest laundering conceal a semantic mutation?

Identify any gate that validates the shape of a declaration rather than independently testing its truth.

---

## 9. Mutation-audit adequacy

Inspect the updated mutation catalogue and execution framework.

Identify the new matrix version and exact mutation count.

At minimum determine whether executable mutations cover:

* splice output replaced with a valid prototype;
* splice decoding to a valid row;
* removal of primary critical contribution;
* removal of secondary critical contribution;
* corruption laundered into a valid prototype;
* control collapsed to one hidden history;
* control collapsed to one hidden label;
* control pixels made non-identical;
* hidden information leaked into provider-visible input;
* final provenance omitting the last intervention;
* temporal-negative episode relabelled valid;
* frame-invalid episode given an incorrect disposition.

For every mutation, inspect whether:

* only the intended property changes;
* immediate digests are recomputed;
* enclosing digests are recomputed when appropriate;
* the mutation is detected through semantic reconstruction;
* the expected primary failure code is stable;
* duplicate mutation effects are rejected;
* repeated audit execution is deterministic.

A mutation detected only because a stale outer digest was left unchanged is weak evidence.

---

## 10. Determinism and causal identity

Verify repeated generation and verification.

Check whether changing:

* root seed;
* source pair;
* source order;
* family version;
* mask/construction identity;
* hidden history;
* intervention sequence;
* observer model;

changes the appropriate identities.

Check whether changing irrelevant filesystem order or Python hash randomization changes output.

Run generation twice in clean temporary directories and compare:

* plans;
* observations;
* family outputs;
* provenance;
* verification reports;
* mutation reports.

Report byte-level or digest-level equality.

---

## 11. Scope discipline and status honesty

Search the branch for status declarations and documentation.

Verify that it promotes only:

```text
episode_family_semantics_correct
```

It must retain:

```text
reference_instrument_correctness_unresolved
prospective_materialization_prohibited
```

It must not imply that the branch establishes:

* action-conditioned reachability;
* actual-action composition;
* provider conformance;
* P3 validity;
* optimized equivalence;
* calibration;
* selection;
* candidate-set utility;
* materialization readiness;
* final evaluation.

Search for success-shaped phrases in:

* status files;
* claim registries;
* generated README files;
* result JSON;
* test names;
* example output;
* CLI messages.

Report any status that exceeds the evidence.

---

# Known out-of-scope defects

Do not mark the branch as failed merely because these broader defects remain, provided the branch reports them honestly:

1. reachability tile v1 may remain action-blind;
2. temporal composition may still use policy-prescribed rather than actual action;
3. final access may remain incompletely guarded at lower layers;
4. provider conformance is not yet independently established;
5. prospective P3 identity remains unresolved;
6. optimized scoring remains absent;
7. materialization remains prohibited.

However, report any change in this branch that worsens, conceals, or falsely closes those defects.

---

# Required tests

Run the branch’s reported tests.

At minimum:

```bash
pytest -q \
  tests/test_video_action_set_family_semantics.py \
  tests/test_video_action_set_family_reachability.py \
  tests/test_video_action_set_benchmark.py \
  tests/test_video_action_set_instrument.py
```

```bash
pytest -q \
  tests/test_video_action_set_reference_verification.py \
  -m "not slow"
```

```bash
pytest -q \
  tests/test_video_action_set_reference_verification.py \
  -m slow --run-slow --durations=20
```

Run the default repository suite:

```bash
pytest -q
```

Also create independent diagnostic scripts where needed.

Do not assume that passing project tests establishes correctness.

---

# Adversarial experiments to perform

Please attempt at least the following.

## Experiment A — Third-state collision

Take a generated frame-invalid record and replace its pixels with a canonical prototype belonging to neither source.

Recompute immediate and enclosing digests where practical.

Expected result:

```text
invalid_family_valid_state_collision
```

## Experiment B — Same-source control

Change all members of an information-control group to one hidden causal history while retaining byte-identical visible input.

Recompute digests.

Expected result:

```text
control_ambiguity_absent
```

## Experiment C — Visible-history leak

Expose a hidden-history identifier through a field available to the tested observer while keeping the pixels identical.

Expected result:

a typed hidden-information leakage failure.

## Experiment D — Provenance truncation

Remove the last operation from a stale-repeat provenance chain while retaining the final emitted pixels and recomputing stored digests.

Expected result:

```text
final_observation_provenance_mismatch
```

## Experiment E — Temporal disposition corruption

Mark a temporal-negative episode as valid while preserving valid individual frame dispositions.

Expected result:

a specific episode-disposition failure.

## Experiment F — Family bypass

Attempt to call lower-level family materializers or constructors directly and determine whether the repaired semantic invariants still execute.

Report any route that can generate an invalid family output without the closure check.

---

# Review output format

Return a structured report.

## A. Reviewed state

Include:

* repository URL;
* branch;
* exact HEAD SHA;
* base SHA;
* commit sequence;
* diff statistics;
* environment;
* Python version;
* dependency installation method;
* test commands and results.

## B. Executive ruling

Choose one:

```text
episode_family_semantics_correct
episode_family_semantics_correct_with_minor_repairs
episode_family_semantics_unresolved
episode_family_semantics_invalid
```

Explain the ruling in no more than five paragraphs.

## C. Confirmed repairs

List only repairs verified through code inspection and/or execution.

## D. Findings ranked by severity

Use:

```text
BLOCKER
CRITICAL
MAJOR
MODERATE
MINOR
```

For every finding include:

* exact file and function;
* failure mechanism;
* reproduction;
* scientific consequence;
* smallest honest repair;
* required regression test.

## E. Measurement table

Report exact pre/post or current counts for:

* frame-invalid records;
* canonical collisions;
* valid-row decodes;
* source equalities;
* information-control groups;
* hidden histories per group;
* visible payloads per group;
* leakage failures;
* provenance replay results;
* mutation declared/executable/detected/missed counts.

## F. Mutation-audit ruling

State whether the mutation audit proves:

```text
artifact integrity only
family semantic validity
both
neither
```

Explain any mutations detected only through stale hashes or collateral changes.

## G. Status ruling

List:

```text
supported statuses
unsupported statuses
statuses that must remain unresolved
```

## H. PR readiness

Choose one:

```text
ready for draft PR
ready for merge after minor corrections
requires another bounded repair commit
not suitable for merge
```

## I. Shortest correction path

If defects remain, provide the smallest ordered correction plan.

Do not propose unrelated features.

---

# Review philosophy

Please be adversarial but evidence-driven.

Do not reject the work merely because it is complex.

Do not accept it merely because it has many schemas, hashes, tests, mutation cases, or green reports.

Focus on the decisive scientific properties:

> Is a frame labelled invalid genuinely outside the valid universe?

> Does a control labelled information-theoretic contain genuinely different hidden causes with identical observer-visible evidence?

> Can the final emitted observation be reconstructed from its declared provenance?

> Can the verifier detect violations after self-consistent digest laundering?

The prior failure was not a lack of verification effort.

It was verification aimed at the wrong property.

The purpose of this review is to determine whether this branch has finally moved the checks from source-relative declarations to domain-level semantic invariants.
