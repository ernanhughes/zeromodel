+++
date = '2026-07-18T01:18:02+01:00'
draft = true
title = 'ZeroModel Visual Research'
+++

## 1. Current position

ZeroModel already has a strong compiled-policy core.

The current architecture can:

* compile a bounded policy into an identity-bearing VPM artifact;
* address that policy using an exact symbolic state;
* return the selected action and complete candidate values;
* trace the decision to an exact policy row and artifact cell;
* reject unknown symbolic states;
* validate the complete finite policy exhaustively;
* serialize and reload the policy without changing its identity;
* attach verification reports, counterexamples and repair lineage.

The first arcade policy contains:

* 112 declared states;
* four candidate actions;
* 448 state-action values;
* exhaustive equivalence across all 2,401 four-target waves.

The policy compilation and lookup layer is therefore not the current problem.

The unresolved problem is the stage before policy lookup:

```text
visual observation
    ↓
which policy state is visibly supported?
    ↓
compiled policy lookup
```

ZeroModel implemented a deterministic Visual Sign Reader that can map canonical rendered frames to the exact policy rows.

That exact canonical path works.

The later held-out benchmark tested whether approximate whole-image representations could tolerate benign visual variation while rejecting invalid observations.

The strongest learned system, based on DINOv2, produced reasonable action accuracy but poor exact-row recovery and unsafe rejection behaviour.

A large proportion of its correct actions came from the wrong policy row.

This means the learned system often recovered an action-equivalent state rather than the actual governed state.

The project therefore needs to revise the observation architecture rather than add more global image-retrieval models.

---

## 2. Relationship to VIPER

VIPER and the proposed visual work operate on opposite sides of the policy.

VIPER performs policy extraction:

```text
powerful teacher policy
    ↓
structured executable policy
    ↓
verification and runtime execution
```

ZeroModel performs a related policy-compilation step:

```text
policy producer
    ↓
scored state-action surface
    ↓
identity-bearing VPM policy artifact
```

The new visual research concerns observation compilation:

```text
image or sensor observation
    ↓
evidence about the current situation
    ↓
state claims
    ↓
compiled policy address
```

Therefore the proposed observation component should not be described as another policy compiler.

A better name is:

# Visual Evidence Compiler

Its purpose is not to create the policy.

Its purpose is to establish what the observation supports before the existing policy is addressed.

---

## 3. Core architectural change

The current learned visual approach attempts:

```text
whole image
    ↓
global embedding
    ↓
nearest known policy row
```

This has several problems:

* a nearest neighbour always exists;
* semantic similarity does not guarantee state identity;
* small local evidence may be discarded;
* two different states may share the same action;
* the system can return the right action from the wrong row;
* invalid observations may still resemble valid states;
* uncertainty becomes hidden inside one similarity score.

The proposed architecture is:

```text
ImageObservation
    ↓
Visual Evidence Compiler
    ↓
factor-level evidence
    ↓
StateClaimSet
    ↓
compatible policy rows
    ↓
declared acceptance contract
    ↓
exact policy lookup or rejection
```

The visual system should no longer be required to force one state prediction.

Instead, it should return the set of state claims that the observation genuinely supports.

---

## 4. Replace `TypedObservedState` with `StateClaimSet`

A single predicted state implies more certainty than the perception system may possess.

Instead of:

```text
tank_position = 3
target_position = 5
cooldown = 0
```

the observation system may need to return:

```text
tank_position ∈ {3}
target_position ∈ {4, 5}
cooldown ∈ {0}
```

Or:

```text
target_present ∈ {true, false}
```

The output should therefore be a `StateClaimSet`.

A `StateClaimSet` represents:

* values directly supported by visual evidence;
* alternative values that remain possible;
* values excluded by evidence;
* ambiguous factors;
* missing evidence;
* structural contradictions;
* impossible combinations.

The policy layer can then determine which rows remain compatible with the observation.

---

## 5. Factorized visual evidence

The next visual system should not predict a complete policy row directly.

It should establish independent policy-relevant factors.

For the arcade fixture, these factors include:

```text
tank_present
tank_count
tank_position
target_present
target_count
target_position
cooldown_indicator_present
cooldown_value
frame_structurally_valid
```

Each factor should produce a structured result containing:

```text
field name
candidate value or value set
status
score or calibrated confidence
evidence region
provider identity
calibration identity
rejection reason
```

Example:

```text
field:
    target_position

candidate values:
    {4, 5}

status:
    ambiguous

evidence region:
    x=16..23, y=1..5

provider:
    sha256:<provider identity>

calibration:
    sha256:<calibration identity>
```

This creates local and inspectable failure explanations.

Instead of saying:

> The image was assigned to the wrong row.

ZeroModel could say:

> Tank position was established correctly, cooldown was established correctly, but target position remained ambiguous between columns four and five.

---

## 6. Concept-bottleneck architecture

This factorized approach is closely related to Concept Bottleneck Models.

The architecture becomes:

```text
image
    ↓
explicit concepts
    ↓
declared state
    ↓
policy decision
```

ZeroModel’s version should be stricter than an ordinary concept bottleneck.

The policy must receive only declared, typed state fields.

It should not receive:

* the original image embedding;
* a hidden residual feature vector;
* an undeclared latent representation;
* a soft vector that secretly contains additional information;
* a direct image-to-action shortcut.

Otherwise the apparent concepts become labels placed over an opaque model rather than a real bottleneck.

The initial implementation should enforce:

```text
no residual path
no hidden embedding path
no joint image-to-action bypass
no undeclared concept dimensions
```

---

## 7. State equivalence must become explicit

The current benchmark assumes that success means recovering the exact policy row.

However, several weaker equivalence relations are possible.

Two states may be:

### Exact-state equivalent

Every declared state field is identical.

### Policy-row equivalent

They resolve to the same compiled row.

### Candidate-vector equivalent

Their complete action-value vectors are identical or sufficiently close.

### Action equivalent

They select the same winning action.

### Property equivalent

They satisfy the same checked policy properties.

### Safety equivalent

They permit the same safe action set.

These equivalence levels must not be mixed.

The DINOv2 result illustrates why:

```text
correct action
    ≠
correct policy row
```

A future ZeroModel reader may support several explicitly named addressing modes, but it must never silently replace exact-state addressing with action-equivalent classification.

Possible modes:

```text
strict_exact
policy_row_exact
value_equivalent
action_safe
property_safe
```

Each mode would have different evidence requirements and different claims.

---

## 8. Build a state-equivalence atlas

Before implementing another vision model, ZeroModel should analyze the existing 112 policy rows.

For every pair of rows, compute whether they are:

* exact-state equivalent;
* action equivalent;
* complete-value-vector equivalent;
* margin equivalent;
* criticality equivalent;
* property equivalent;
* safety equivalent.

The output should be a state-equivalence atlas.

Then reclassify all existing visual benchmark errors.

For each wrong-row prediction, determine whether it was:

```text
wrong row, same action
wrong row, same complete value vector
wrong row, same safety properties
wrong row, different margin
wrong row, conflicting action
wrong row, unsafe policy conflict
```

This analysis will show what representation the existing visual providers actually learned.

It may reveal that a provider captured a coarse action abstraction while failing exact identity.

That does not turn the failed experiment into a success.

It tells us precisely what was preserved and what was lost.

---

## 9. Inverse perception contracts

The visual system should return uncertainty sets rather than overconfident point estimates.

Instead of:

```text
target_position = 5
```

return:

```text
target_position ∈ {4, 5}
```

with a declared calibration or coverage guarantee.

The assembled observation contract may then identify a set of compatible policy rows:

```text
compatible rows:
    row_41
    row_42
```

The policy layer can inspect those rows.

Possible outcomes:

### Exact singleton

Only one policy row remains possible.

```text
accept exact address
```

### Action-consistent ambiguity

Several rows remain possible, but every row selects the same action.

```text
optionally accept under action_safe mode
```

### Value-consistent ambiguity

Several rows remain possible, but their complete candidate vectors and safety properties are equivalent under a declared tolerance.

```text
optionally accept under value_equivalent mode
```

### Conflicting ambiguity

The compatible rows select different actions or have different safety properties.

```text
reject or escalate
```

This is more principled than choosing the nearest row and inspecting its distance.

---

## 10. Policy-row candidate sets

The output of observation processing should therefore be:

```text
StateClaimSet
    ↓
set of compatible policy rows
```

Example:

```text
tank_position = {3}
target_position = {4, 5}
cooldown = {0}
```

This could produce:

```text
compatible policy rows:
    tank=3|target=4|cooldown=0
    tank=3|target=5|cooldown=0
```

ZeroModel then evaluates:

* whether the candidate set is a singleton;
* whether all candidates choose the same action;
* whether all candidates satisfy the same safety properties;
* whether their decision margins differ materially;
* whether criticality differs materially;
* whether any candidate is forbidden.

The visual provider does not decide whether ambiguity is acceptable.

The policy contract decides.

---

## 11. Risk-calibrated rejection

The existing benchmark uses thresholds and margins derived from similarity scores.

The next system should calibrate rejection against explicit operational risks.

Potential errors have different consequences.

A possible loss scale:

```text
0:
correct exact row

1:
wrong row but identical complete action values

2:
wrong row with same winning action but different margin

5:
wrong row but same declared safety properties

20:
wrong row with conflicting action

100:
critical evidence absent but observation accepted
```

Calibration should optimize these declared risks rather than raw classification accuracy.

Important metrics include:

```text
exact singleton rate
ground-truth row coverage
mean candidate-set size
unsafe singleton rate
conflicting-action acceptance
critical-evidence false acceptance
false rejection rate
risk-weighted loss
```

This would make acceptance behaviour directly related to the policy’s consequences.

---

## 12. Conformal or set-valued calibration

Conformal methods may be useful for producing calibrated candidate sets.

The goal would not merely be:

> Include the correct state 95% of the time.

The goal might be:

```text
unsafe singleton acceptance below a declared bound
conflicting-action acceptance below a declared bound
critical-evidence omission below a declared bound
```

Per-factor conformal sets could produce:

```text
tank_position ∈ {3}
target_position ∈ {4, 5}
cooldown ∈ {0}
```

The row assembler could then compute the Cartesian set of compatible policy rows.

The calibration artifact should be independently identified and linked to:

* the visual provider;
* the factor schema;
* the benchmark dataset;
* the environment;
* the policy artifact;
* the declared risk target.

---

## 13. Local evidence instead of global scene similarity

The DINOv2 CLS-token baseline uses a global representation.

Global representations may suppress exactly the distinctions needed for governed addressing.

For example:

* one-pixel position changes;
* missing cooldown indicators;
* an extra object;
* object count;
* local structural validity.

The next providers should operate on local evidence.

Candidate providers include:

```text
connected-component extraction
geometric rules
local normalized-pixel templates
patch matching
small per-factor classifiers
DINOv2 patch-token probes
compact multi-head task-specific models
```

The unit of comparison should become:

```text
provider quality for one declared factor
```

rather than:

```text
whole-image accuracy for the final policy row
```

---

## 14. Task-aware representations

If learned visual providers are added, they should be trained around policy-relevant distinctions.

Possible supervision targets include:

```text
same tank position
same target position
same cooldown
same structural validity
same policy row
same action-value vector
same safe-action set
same checked properties
```

This is different from using a general semantic embedding.

A task-aware representation should be evaluated on whether it preserves the information required by the policy contract.

A large or semantically powerful representation is not automatically control-sufficient.

---

## 15. Prevent concept leakage

Soft concepts can accidentally encode more information than their names imply.

For example, a continuous vector labelled `target_position` may also contain:

* tank position;
* cooldown state;
* action preference;
* background corruption;
* row identity.

This would make the concept layer difficult to interpret or intervene upon.

ZeroModel should initially prefer:

* typed categorical values;
* finite candidate sets;
* explicit missing and ambiguous states;
* declared evidence regions;
* hard interfaces between factors and policy;
* no hidden continuous bypass.

Learned providers may internally use complex representations.

Their exported contract must remain explicit.

---

## 16. Temporal belief is a separate capability

Some states cannot be determined from one frame.

The existing target-removal control demonstrates this.

A frame in which a target has been removed may be visually identical to a legitimate no-target state.

No single-frame representation can reconstruct the hidden history.

This is an information problem, not merely a model-quality problem.

ZeroModel should distinguish:

```text
single-frame evidence
```

from:

```text
temporal belief state
```

A temporal belief component may combine:

```text
previous state claims
current visual evidence
previous action
legal transition model
object persistence
elapsed time
```

It may maintain several hypotheses:

```text
target genuinely absent
target temporarily occluded
target removed unexpectedly
frame corrupted
```

Temporal reasoning should not be hidden inside the visual provider.

It should be a separately identified component with its own trace.

---

## 17. Explicit fallback architecture

When observation evidence cannot support a safe policy address, the system must not force a decision.

Possible fallbacks include:

```text
safe-stop action
verified baseline controller
direct symbolic instrumentation
human review
larger perception model
general planning model
request another observation
```

The fallback system should be explicit and identified.

A complete deployment trace should contain:

```text
primary policy artifact
visual evidence provider
calibration artifact
state claim set
candidate policy rows
acceptance or rejection rule
fallback policy artifact
selected fallback reason
final action
```

This creates a Simplex-style architecture:

```text
high-performance compiled path
        +
safe fallback path
        +
identified switching contract
```

---

## 18. Direct instrumentation must remain a baseline

In a bounded digital system, direct access to the real state may be safer and cheaper than visual inference.

The benchmark should always compare:

```text
direct engine or sensor state
deterministic visual extraction
learned visual extraction
```

The purpose of visual observation is not to replace direct state access where state access is already reliable.

Visual addressing is most relevant when the observer only has access to:

* rendered output;
* camera feeds;
* legacy interfaces;
* recorded frames;
* independent monitoring channels.

The direct-instrumentation baseline prevents ZeroModel from solving a harder problem unnecessarily.

---

## 19. New artifact types

The enhanced architecture may require separate artifacts for:

### Factor schema artifact

Declares:

```text
factor names
types
allowed values
required fields
structural constraints
```

### Provider manifest

Declares:

```text
provider identity
implementation version
input contract
output factor
preprocessing
model or rule identity
```

### Evidence bundle

Contains:

```text
observation digest
factor claims
scores
evidence regions
rejection reasons
```

### Calibration artifact

Contains:

```text
calibration dataset identity
risk target
thresholds or set construction
coverage statistics
```

### StateClaimSet artifact

Contains:

```text
candidate values per factor
compatible policy rows
ambiguity structure
```

### Acceptance-contract artifact

Declares:

```text
exact-state mode
action-safe mode
value-equivalent mode
safety requirements
rejection rules
```

### Temporal belief artifact

Contains:

```text
prior belief
new evidence
transition constraints
updated hypotheses
```

### Fallback-binding artifact

Links:

```text
primary policy
fallback policy
switching contract
```

This continues ZeroModel’s central discipline: meaningful boundaries should be explicit, identified and traceable.

---

## 20. Revised runtime pipeline

The proposed full runtime path is:

```text
raw observation
    ↓
canonical observation contract
    ↓
factor-level evidence providers
    ↓
EvidenceBundle
    ↓
per-factor calibrated candidate sets
    ↓
StateClaimSet
    ↓
state-consistency checks
    ↓
compatible policy-row set
    ↓
declared equivalence contract
    ↓
accept exact / accept safe ambiguity / reject
    ↓
VPMPolicyLookup
    ↓
action + complete trace
```

For temporal systems:

```text
previous belief
    +
current StateClaimSet
    +
transition contract
    ↓
updated temporal belief
    ↓
compatible policy-row set
```

---

## 21. Proposed experiment 1: State-equivalence atlas

Do not add a new visual model yet.

First analyze all 112 policy rows.

Compute:

```text
exact state clusters
winning-action clusters
complete-value-vector clusters
decision-margin clusters
criticality clusters
property-equivalence clusters
safe-action clusters
```

Re-score existing benchmark predictions under each relation.

Questions:

* What abstractions did DINOv2 preserve?
* What abstractions did normalized pixels preserve?
* Which wrong rows were harmless under a weaker contract?
* Which wrong rows were unsafe?
* Is exact row identity required for every deployment mode?

Deliverable:

```text
StateEquivalenceReport
```

---

## 22. Proposed experiment 2: Deterministic factor baseline

Implement a hard visual concept bottleneck using simple methods.

Providers:

```text
connected components
local pixel templates
geometric extraction
region-specific thresholding
```

Outputs:

```text
tank count and position
target count and position
cooldown presence and value
structural validity
```

Benchmark against whole-image normalized pixels.

Questions:

* Does factorization improve exact row recovery?
* Does it improve rejection of removed evidence?
* Can it reject impossible two-object scenes?
* Which factors fail under translation, brightness and noise?
* Is the contract more useful diagnostically than whole-image retrieval?

---

## 23. Proposed experiment 3: Set-valued state compilation

Calibrate factor providers to return candidate sets.

Example:

```text
target_position ∈ {3, 4}
```

Assemble those into compatible policy-row sets.

Measure:

```text
ground-truth row coverage
singleton exact-address rate
mean candidate-set size
action-consistent ambiguity
conflicting-action ambiguity
unsafe singleton rate
critical-evidence false acceptance
false rejection rate
```

Question:

> Can the system preserve uncertainty without either rejecting everything or accepting unsafe states?

---

## 24. Proposed experiment 4: Learned local providers

Only after deterministic factor extraction is measured.

Compare:

```text
local normalized pixels
template matching
small per-factor CNN
DINOv2 patch-token probe
small multi-head factor model
task-aware contrastive representation
```

The learned components should predict evidence factors, not final policy rows.

Question:

> Does a learned component provide enough improvement on a specific factor to justify its complexity?

---

## 25. Proposed experiment 5: Risk-calibrated acceptance

Construct operating curves for:

```text
exact-row recovery
candidate-set coverage
conflicting-action acceptance
critical-evidence false acceptance
false rejection
risk-weighted loss
```

Compare:

```text
global thresholds
per-factor thresholds
per-state thresholds
conformal candidate sets
risk-optimized rejection
```

Question:

> Can acceptance be calibrated around policy consequences rather than generic similarity?

---

## 26. Proposed experiment 6: Temporal ambiguity

Create sequences containing:

```text
temporary occlusion
target removal
impossible disappearance
duplicate objects
cooldown indicator loss
frame corruption
```

Maintain multiple state hypotheses over time.

Measure:

```text
belief coverage
time to resolve ambiguity
unsafe premature commitment
correct fallback activation
```

Question:

> Can temporal evidence resolve cases that are impossible from one frame without fabricating certainty?

---

## 27. Proposed experiment 7: Fallback and switching

Define:

```text
primary compiled policy
safe fallback policy
switching contract
```

Test:

* exact accepted state;
* benign ambiguity;
* conflicting-action ambiguity;
* invalid frame;
* missing critical evidence;
* provider failure;
* calibration mismatch;
* policy identity mismatch.

Question:

> Does the system always reject or switch safely when its evidence contract is not satisfied?

---

## 28. Proposed experiment 8: Governance-value comparison

Compare the complete ZeroModel chain with a conventional implementation:

```text
ordinary detector
    +
SHA-256 model digest
    +
JSONL structured log
```

Measure:

```text
implementation complexity
runtime overhead
trace completeness
replay fidelity
localized failure explanation
policy-version binding
calibration binding
review usability
repair workflow
```

Question:

> Does ZeroModel’s artifact architecture provide enough additional governance value to justify its complexity?

This is a critical experiment.

The visual project should not assume that more artifacts automatically produce more useful governance.

---

## 29. Metrics that should replace headline action accuracy

Headline action accuracy is insufficient because many policy rows share an action.

The primary metrics should become:

```text
exact state accuracy
exact policy-row accuracy
complete-value-vector equivalence
candidate-set coverage
candidate-set size
action correctness
action correctness from wrong row
conflicting-action error
unsafe acceptance
safe rejection
critical-evidence false acceptance
risk-weighted loss
```

Every report should separate:

```text
correct exact state
correct action from wrong state
safe ambiguity
unsafe ambiguity
rejection
false acceptance
```

---

## 30. Claim boundaries

The enhanced system should not claim:

* general visual understanding;
* open-world perception;
* semantic scene comprehension;
* safe learned addressing without measured evidence;
* exact state recovery from information-theoretically ambiguous frames;
* deployment readiness;
* tolerance to arbitrary cameras, lighting or occlusion;
* formal safety from calibration alone;
* that a concept name guarantees semantic faithfulness;
* that a correct action proves a correct observation address.

The strongest possible future claim would be narrower:

> Within a declared bounded observation domain, ZeroModel can compile calibrated visual evidence into explicit state claims, preserve unresolved alternatives, reject incompatible or unsafe ambiguity, and delegate accepted states to an independently identified compiled policy.

---

## 31. What should remain unchanged

The following current ZeroModel components remain valuable:

```text
VPM policy artifacts
VPMPolicyLookup
stable policy row IDs
complete candidate action values
criticality and decision margins
policy property checking
verification artifacts
counterexample localization
repair and re-verification lineage
content-addressed artifact identity
provider-neutral observation contracts
benchmark family separation
information-theoretic controls
explicit rejection decisions
```

The negative DINOv2 result does not weaken these components.

It shows that the observation provider must be redesigned.

---

## 32. What should not be promoted

Do not promote:

```text
global DINOv2 CLS nearest-neighbour retrieval
DINOv2 medoid retrieval
DINOv2 all-prototype row retrieval
ridge row classifier
whole-image semantic similarity as state identity
action accuracy as sufficient proof
single-threshold rejection as complete safety
```

Keep these implementations as reproducible research baselines.

---

## 33. Recommended implementation sequence

### Stage 1: Close the current evidence

* commit the raw benchmark report;
* commit the environment manifest;
* fix provenance-recording errors;
* add confidence intervals;
* add proper benign denominators;
* preserve per-family results;
* update the claims audit;
* record the stop decision.

### Stage 2: Build the equivalence and failure atlas

* analyze state equivalence;
* generate row-confusion matrices;
* analyze score distributions;
* plot FAR/FRR curves;
* classify action-equivalent errors;
* inspect translation failures;
* test patch-token locality.

### Stage 3: Specify the Visual Evidence Compiler

* define factor schema;
* define evidence result contract;
* define StateClaimSet;
* define candidate-row assembly;
* define equivalence modes;
* define rejection rules;
* define benchmark and kill conditions.

### Stage 4: Implement deterministic factor providers

* direct instrumentation;
* connected components;
* local templates;
* geometry;
* structural validity.

### Stage 5: Add calibrated candidate sets

* per-factor calibration;
* set-valued claims;
* compatible-row computation;
* safe ambiguity rules.

### Stage 6: Add learned factor providers selectively

Only where simple deterministic extraction demonstrably fails.

### Stage 7: Add temporal belief and fallback

Only after distinguishable single-frame cases are handled.

---

## 34. Kill conditions

Stop or narrow the direction if:

1. direct instrumentation is clearly safer, simpler and cheaper;
2. factorized extraction does not improve exact-row recovery;
3. critical-evidence absence cannot be detected reliably;
4. calibrated candidate sets remain too large to be useful;
5. safe rejection requires rejecting nearly every benign observation;
6. learned providers require near-exhaustive examples for every state;
7. the artifact chain adds little value beyond a conventional detector and structured log;
8. synthetic results do not transfer to a fixed-camera bounded environment;
9. concept providers cannot be prevented from leaking undeclared information;
10. the architecture becomes more complex than the operational problem justifies.

---

## 35. Main research contribution being tested

The proposed contribution is not:

> ZeroModel invented visual state extraction.

The proposed contribution is:

> ZeroModel can place a governed, identity-bearing and rejectable contract between perception and a compiled policy.

That contract would make explicit:

```text
what was observed
which evidence providers were used
which state factors were supported
which values remained possible
which policy rows remained compatible
which equivalence contract allowed acceptance
which policy artifact was addressed
which action was selected
why the system rejected or fell back
```

---

## 36. Final synthesis

The first ZeroModel phase compiled policy.

The visual phase attempted to retrieve that policy directly from whole-image similarity.

The learned retrieval experiment showed that semantic visual representations may preserve broad action similarity while losing the exact local distinctions required for state identity and safe rejection.

The next architecture should therefore stop treating perception as one image-to-row guess.

It should:

```text
extract explicit evidence
preserve ambiguity
assemble state claims
identify compatible policy rows
apply a declared equivalence contract
reject unsafe uncertainty
delegate accepted cases to the exact compiled policy
```

The resulting complete architecture is:

```text
Observation
    ↓
Visual Evidence Compiler
    ↓
EvidenceBundle
    ↓
StateClaimSet
    ↓
Compatible Policy Rows
    ↓
Identity / Value / Safety Contract
    ↓
Exact VPM Policy Lookup
    ↓
Action + Complete Trace
```

VIPER remains the precedent for compiling a policy from a powerful source.

The new ZeroModel research asks a different question:

> Can the system establish which real-world situation that compiled policy is being asked to govern, while preserving uncertainty and refusing to invent a state that the evidence does not support?


## The Simplest Remaining Explanation

The DINOv2 experiment gave us a clear negative result.

A global semantic embedding could often recover the correct action, but it could not reliably recover the exact policy row or reject observations that should never have been accepted.

Our first interpretation was architectural.

Perhaps whole-image retrieval was the wrong abstraction. Perhaps the visual reader needed to decompose the observation into explicit evidence: object positions, indicators, geometry, counts and other factors that could be assembled back into a policy address.

That direction may still be correct.

But the external reviewers raised a more immediate objection:

> Before building a Visual Evidence Compiler, had we actually exhausted the simplest possible visual representation?

The strongest untested candidate was not another foundation model.

It was the pixels themselves.

---

### Normalized Pixels as a Serious Baseline

System B represents each observation using the complete grayscale pixel array.

The transformation is deliberately small:

```text
rendered observation
    ↓
grayscale pixel vector
    ↓
subtract the observation mean
    ↓
L2-normalize the vector
    ↓
compare against canonical policy-row prototypes
```

There is no learned encoder.

There are no semantic features.

There is no object detector, neural network or language model.

The representation preserves the exact local correspondence that DINOv2’s global embedding appeared to discard.

This made System B an unusually important test.

If calibrated normalized pixels produced a safe operating region, we would already possess a useful bounded visual reader. Building a more elaborate factorized perception architecture would then be premature.

If normalized pixels failed, however, the nature of that failure would tell us what the next architecture actually needed to repair.

The reviewers were right about the sequence.

We therefore stopped designing the larger system and returned to the baseline.

---

### Ranking Is Not Acceptance

The earlier benchmark had already shown that normalized pixels contained substantial signal.

At a permissive operating point, System B frequently ranked the correct row near the top and selected the correct action even when it retrieved the wrong row.

But this did not establish that it was safe to use.

A visual reader has two separate responsibilities:

```text
1. rank the possible policy rows;
2. decide whether the best-ranked row is trustworthy enough to execute.
```

Those are not the same problem.

A representation may be good at ranking while being unable to distinguish:

```text
a correct match
from
a plausible but wrong match
```

This distinction matters enormously in ZeroModel.

The output is not merely a classification label. It is an address into a compiled policy artifact.

Accepting the wrong row can silently attach an action to the wrong state, the wrong candidate values, the wrong criticality, the wrong verification properties and the wrong provenance.

A reader that is usually correct but cannot identify its own mistakes is not a governed address reader.

It is a forced nearest-neighbour lookup.

The purpose of Phase A was therefore not to make normalized pixels win.

It was to determine whether System B possessed any operating point at which it could accept useful observations without also accepting distinguishably invalid ones.

---

### Repairing the Evaluation Boundary

This required a stricter evaluation protocol than the original benchmark.

The observations were divided by role:

```text
canonical prototypes
benign calibration observations
rejection calibration observations
untouched final evaluation observations
information-theoretic impossibility controls
```

These categories could not be treated interchangeably.

Canonical prototypes define the policy-row codebook.

Benign calibration observations represent transformations the reader may reasonably be expected to tolerate.

Rejection observations are visually distinguishable cases that must not be mapped onto a policy row.

The final evaluation set remains unavailable during threshold selection.

Information-theoretic controls represent cases where the supplied observation no longer contains enough evidence to recover the original state. They are useful for identifying the limits of the task, but they must not be counted as ordinary false accepts or false rejects.

That last distinction was especially important.

A visually corrupted observation and an observation from which the required information has been completely removed are not equivalent failures.

One tests the reader.

The other tests whether the answer still exists in the input.

The repaired protocol therefore excluded impossibility controls from the ordinary benign, rejection, false-accept and false-reject denominators. They remained visible as a separate diagnostic property.

We also separated the exploratory global threshold sweep from the actual calibration procedure.

A curve computed using final-evaluation traces may help us understand the score distribution, but it cannot legitimately select a deployment threshold. Doing so would allow the test set to influence the system being tested.

The global sweep was retained, but explicitly labelled:

```text
exploratory
not equivalent to per-row calibration
uses final traces
invalid for threshold selection
```

This may sound procedural, but it is central to the research.

A threshold is part of the reader.

Selecting it on the final test set is another form of training on the answer.

---

### The Operating-Curve Result

System B was evaluated across the complete declared quantile grid.

The selected point was:

```text
quantile = 1.0
```

At that point, the final raw ranking results were:

```text
exact row ranked first:     1,008 / 1,344
correct action ranked first: 1,302 / 1,344
```

That corresponds to:

```text
75.0% exact-row ranking accuracy
96.875% action-ranking accuracy
```

The representation was clearly not empty.

It preserved a large amount of policy-relevant structure.

On most benign observations, normalized pixels placed the correct action above the alternatives. On three quarters of them, it placed the exact policy row first.

But the acceptance result was radically different:

```text
accepted benign observations: 0 / 1,344
correct accepted rows:         0 / 0
distinguishable false accepts: 0 / 248
false rejections:              1,344 / 1,344
```

The threshold eliminated all observed distinguishable false accepts.

It also rejected every benign final observation.

There was no accepted precision to calculate because there were no accepted observations.

This is **Outcome C**:

> System B retains useful pre-rejection ranking signal, but it has no useful zero-distinguishable-false-accept operating region on the corrected bounded arcade fixture.

The result is not that normalized pixels cannot recognize the scene.

The result is that their similarity scores do not provide a reliable boundary between correct and unsafe retrieval.

---

### The Reader Usually Knows the Action—but Does Not Know When It Knows

This was the most interesting result of the phase.

System B selected the correct action for almost 97% of the benign final observations before rejection.

That sounds close to success.

But when required to separate safe matches from unsafe ones, the only threshold that eliminated observed distinguishishable false accepts rejected everything.

The system frequently knew which action looked best.

It did not know whether the observation justified acting on that belief.

This gives us a more precise description of the problem:

```text
the representation contains decision signal
but its confidence geometry does not support governed acceptance
```

That is a different failure from the DINOv2 result.

The global DINOv2 representation lost too much exact local structure.

Normalized pixels preserved far more local structure, but their whole-image distance remained too sensitive to benign variation and too poorly separated from invalid observations.

The best match could be useful without being safely distinguishable.

This suggests that the missing component is not necessarily more semantic intelligence.

It may be **local correspondence**.

A target shifted by one cell, a missing indicator or a small structural corruption should not be reduced to one global distance over the entire frame. The reader may need to determine which local regions agree, which disagree and whether the disagreement affects the policy address.

---

### A Negative Result That Narrows the Architecture

Phase A ruled out two tempting shortcuts.

The first was:

```text
use a powerful global visual embedding
```

The second was:

```text
use direct whole-image pixel similarity
```

Neither produced a useful governed operating region on the current fixture.

But they failed differently.

DINOv2 frequently erased distinctions that were crucial to exact policy identity.

Normalized pixels retained those distinctions in their rankings but could not turn the resulting global distances into a safe acceptance boundary.

That difference directs the next experiment.

The next reader should preserve local geometry without reducing the entire observation to either:

```text
one global semantic vector
```

or:

```text
one global pixel-distance score
```

The smallest justified candidates are therefore conventional and deterministic:

```text
bounded registration
local template correlation
connected components
region-specific geometry extraction
explicit indicator detection
```

These methods can answer questions that a whole-image score cannot:

```text
Was the tank found in the expected region?

How many targets were detected?

Which target columns are occupied?

Is the cooldown indicator present?

Did one local component move, disappear or duplicate?

Which exact piece of evidence caused rejection?
```

Only after these baselines are measured will a learned local representation or a full factorized visual provider be justified.

---

### The Evidence Chain Also Had to Survive Review

The implementation produced a credible numerical Outcome C, but the branch review identified several weaknesses in the evidence package around it.

One recovered runtime log was listed in the recovery manifest but was not actually present in the committed directory.

An undefined `0 / 0` accepted precision was correctly represented as `null`, while its confidence interval was incorrectly serialized as `[0, 0]`.

The environment record captured Python, NumPy, PyTorch, CUDA and GPU information, but did not yet bind the run to the exact Git commit, branch, dirty state, command line and evidence digests.

The candidate-selection API also encoded final-evaluation observations even though the current pixel encoder did not use batch-level fitting. This did not appear to alter the measured result, but the boundary should be enforced rather than merely respected by the present implementation.

None of these issues provides evidence that System B secretly possesses a useful operating region.

They do establish something equally important:

> A negative result needs the same identity, lineage and reproducibility discipline as a successful artifact.

The result should not be promoted merely because its conclusion is convenient.

Before the next visual experiment becomes the new research branch, the Phase A evidence must be closed under the same standard ZeroModel applies to policy artifacts:

```text
exact files
exact digests
exact source commit
exact command
exact dataset identity
exact selection identity
exact metric semantics
```

That corrective work is small, but it is not optional.

---

### What We Can Honestly Claim

After Phase A, the strongest supported claims are narrow:

1. Global DINOv2 CLS embeddings do not provide safe exact visual addressing on this bounded arcade fixture.

2. Mean-centred, L2-normalized pixels preserve substantial exact-row and action-ranking signal.

3. That ranking signal does not translate into a useful zero-observed-distinguishable-FAR acceptance region under the corrected calibration protocol.

4. Correct action ranking is not equivalent to correct policy addressing.

5. A visual reader must be evaluated not only on what it selects, but on whether it can recognize when selection is unjustified.

6. The next justified experiment is local deterministic correspondence, not a larger semantic model.

We cannot yet claim that semantic representations are inherently incompatible with governed identity.

We cannot claim that normalized pixels contain no useful information.

We cannot claim that factorized perception is required.

And we cannot yet claim that a visual observation can safely invoke the compiled policy under benign variation.

What we have gained is more valuable than a premature success.

We now know more precisely where the unresolved problem lives:

```text
not merely in recognition
not merely in retrieval
but at the boundary between evidence and justified execution
```

That boundary is the next object of the research.
