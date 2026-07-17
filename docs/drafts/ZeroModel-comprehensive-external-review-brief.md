# ZeroModel: Comprehensive Project Brief and External Review Prompt

## Purpose

This document is intended for independent review by multiple AI systems. The goal is not encouragement or a generic summary. The goal is to identify what ZeroModel actually is, what is technically distinctive, what is established engineering, where the strongest research contribution lies, what is missing, and what should be built next.

Review the project critically but constructively. Separate:

1. defects or contradictions;
2. missing evidence;
3. useful extensions;
4. unnecessary complexity;
5. adjacent prior art;
6. genuine new research opportunities.

Do not reduce ZeroModel to only one component such as a Q-table, heatmap, cache, decision table or policy reader. Those are relevant precedents and partial descriptions, but the project claims a broader artifact architecture. At the same time, do not accept novelty merely because familiar mechanisms have been combined. Evaluate whether the combination creates a coherent, useful and testable system abstraction.

---

# 1. Project in One Sentence

**ZeroModel turns scored data into deterministic, inspectable Visual Policy Map artifacts and small consumers that can operate without a model at decision time.**

A Visual Policy Map, or VPM, is not simply an image. It is a machine-readable artifact containing:

- a source matrix of scored rows and metrics;
- stable row identifiers;
- stable metric identifiers;
- an explicit spatial layout recipe;
- deterministic row and column ordering;
- normalized view values;
- mappings from view coordinates back to source coordinates;
- provenance and parent relationships;
- deterministic content identity;
- optional PNG or SVG renderings;
- lossless `.vpm` serialization.

Different consumers can use the same artifact for lookup, inspection, comparison, rendering, verification, gating, learning traces, training telemetry, critic evidence or temporal analysis.

Repository: https://github.com/ernanhughes/zeromodel

Project site: https://zeromodel.org

Technical article: https://programmer.ie/post/zeromodel/

---

# 2. Core Thesis: Compiled Versus Invoked

Most current AI systems are organized around repeated invocation:

```text
current state
        ↓
invoke model, agent or planner
        ↓
reconstruct a decision
        ↓
execute action
```

ZeroModel asks whether every decision should remain inside that loop.

Some decisions are genuinely novel, ambiguous or open-ended. Those still require models, planning, search, tools or human judgment.

Other decisions eventually become:

- bounded;
- stable;
- reviewed;
- repeated;
- addressable;
- policy-like.

For those regions, ZeroModel proposes a compiled architecture:

```text
model, expert, optimizer,
training process or rules
        ↓
scored policy or evidence surface
        ↓
compile deterministic VPM artifact
```

Later, at runtime:

```text
current state
        ↓
resolve stable address
        ↓
read prepared candidate values
        ↓
select action
        ↓
return action + artifact trace
```

The intelligence has not disappeared. It has moved upstream.

The governing metaphor is:

> **Use intelligence for novelty. Build signs for stable policy.**

Or more precisely:

> **Do not keep invoking intelligence merely because the system has never materialized what that intelligence already decided.**

This is the meaning of “Signs, Not Directions.”

---

# 3. Problems ZeroModel Is Trying to Solve

## 3.1 Policy remains implicit

A policy may be distributed across model weights, prompts, tools, scripts, orchestration code, retrieval sources, institutional memory and human approval. A system can produce decisions without possessing a durable object representing the stable policy behind them.

ZeroModel asks whether stable policy can become a first-class system object that can be identified, serialized, inspected, compared, versioned, verified, replayed, promoted and retired.

## 3.2 Runtime traces are often too weak

A conventional action log may say:

```text
action = FIRE
```

A ZeroModel policy decision can retain:

```text
artifact ID
state row ID
selected action
selected value
all candidate values
source row index
source metric index
view row
view column
optional evidence metrics
```

This does not explain why the upstream policy generator assigned its values. It does establish exactly which compiled artifact and cell supplied the runtime action.

## 3.3 Alternatives are commonly discarded

Many systems retain only the selected output. ZeroModel preserves the complete candidate vector. This makes it possible to inspect not only what won, but what lost, how close the decision was, how consequential a poor choice might be, and whether a policy mutation changed the winner or only changed the surrounding evidence.

## 3.4 Stable work is repeatedly recomputed

Software already avoids recomputing stable work through compilation, caching, indexes, routing tables, materialized views, memoization, residual programs and precomputed policies.

ZeroModel applies this staging idea to bounded AI policy and evidence while adding an artifact contract around identity, layout, traceability and inspection.

---

# 4. Core Artifact Architecture

## 4.1 `ScoreTable`

The source abstraction is a table of scored relationships.

Examples:

```text
states × actions
documents × evidence metrics
training checkpoints × progress metrics
critic findings × risk dimensions
incidents × response options
```

The table contains stable row and metric identifiers rather than relying only on numeric array positions.

## 4.2 `LayoutRecipe`

A layout recipe declares how source rows and metrics become a spatial view.

Supported forms include source ordering, lexicographic ordering, weighted-score ordering, explicit column ordering, per-metric normalization and deterministic tie-breaking.

The layout is part of the representation, not an accidental result of a plotting library.

## 4.3 `VPMArtifact`

The compiled artifact contains:

- the original scored source;
- the recipe;
- deterministic source digests;
- row and column order;
- normalized field values;
- source-to-view mappings;
- provenance;
- parents;
- canonical identity.

The project uses canonical numeric encoding and SHA-256 identity. This establishes artifact identity under the canonicalization contract. It does not establish correctness, authorship, approval, authorization, truthful provenance or universal semantic equivalence.

## 4.4 Rendering and serialization

The artifact can be rendered to PNG or SVG for inspection. The image is a view of the artifact, not the complete machine-readable artifact.

The `.vpm` bundle contains the serializable representation required to reconstruct and validate artifact identity.

## 4.5 Multiple deterministic views

The same scored source can support several views:

```text
source-order view
risk-first view
criticality-first view
people-first view
tree-first view
optimized top-left view
```

Each view has a deterministic identity because layout is part of the artifact contract, while source mappings preserve the connection to the same underlying rows and metrics.

---

# 5. Runtime Policy Lookup: “Signs, Not Directions”

`VPMPolicyLookup`, also exposed as `SignReader`, is one consumer of the artifact.

For a finite policy:

- rows are discretized runtime states;
- metric columns are candidate actions;
- cells contain action values;
- the reader performs deterministic argmax over declared action metrics;
- the result contains the winning action and exact artifact/source/view coordinates.

The artifact may also contain evidence columns that are returned with the trace but excluded from action selection.

The deployment convention is:

> **The artifact ID recorded in a runtime decision trace is the identity of the exact artifact consumed by the reader.**

Related artifacts should be connected through provenance rather than treated as interchangeable identities.

---

# 6. Finite Arcade Experiment

The reference experiment is intentionally small enough to enumerate completely.

The world contains:

- a tank in one of seven positions;
- a target in one of seven positions or absent;
- cooldown active or inactive;
- four actions: `LEFT`, `RIGHT`, `STAY`, `FIRE`.

This produces:

- 112 declared states;
- 448 state-action values;
- 2,401 possible ordered four-target waves.

The exhaustive committed validation checks:

- 112 / 112 state rows;
- 448 / 448 action values;
- 112 / 112 selected actions;
- 2,401 / 2,401 possible waves;
- 31,213 matched source-versus-artifact runtime steps;
- persistence and reload;
- deterministic artifact identity;
- decision-trace equivalence;
- exact source and view cell resolution.

The shooter is not evidence of commercial usefulness or open-world intelligence. It is a “hydrogen atom” for the architecture: a complete policy small enough that every declared state and trajectory can be checked rather than sampled.

---

# 7. What Is New in the Current Project

## 7.1 Q-bearing evidence

A Q-bearing policy can add two non-action evidence metrics:

```text
criticality
    = best action value - worst action value

decision margin
    = best action value - second-best action value
```

Criticality asks how costly a poor action could be. Decision margin asks how decisively the winner beat its nearest alternative.

The best-minus-worst value should be called VIPER-style criticality only when the source columns contain Q-values or an equivalent consequence-bearing signal. For arbitrary handcrafted scores, it is only score spread.

The original binary arcade policy has a flat best-minus-worst surface, so the criticality fixture uses a separate Q-bearing teacher with varying consequence gaps.

## 7.2 Action/evidence separation

Evidence columns remain inside the artifact and decision trace but cannot participate in action argmax. This preserves forensic information without allowing diagnostic metrics to become accidental actions.

## 7.3 Declarative finite policy properties

`PolicyPropertyChecker` exhaustively checks named, versioned row-level properties over a finite policy.

Example:

```text
If FIRE wins,
tank must equal target
and cooldown must equal zero.
```

The property language is a closed JSON-style interpreter rather than arbitrary `eval`. It supports variable lookup, equality, inequality, numeric comparison, membership, conjunction, disjunction, negation and material implication.

Typed row IDs decode nulls, booleans, integers, floats and strings deterministically. Evaluation errors include property ID, property version, failing row, operator, operand values and operand types.

## 7.4 Verification artifacts

A property-check report can itself become an identity-bearing VPM artifact.

The verification artifact records:

- checked policy artifact ID;
- checker version;
- property IDs and versions;
- digest of the property specification;
- rows checked;
- pass or fail;
- violations;
- exact counterexample rows;
- selected actions;
- candidate values;
- evidence;
- source and view coordinates.

Its provenance contains a `verifies` parent relation pointing to the exact checked policy.

This supports the precise claim:

> **This identified finite policy artifact passed these named row-level properties under this checker and property specification.**

It does not prove that the policy is universally safe or that the property set is sufficient.

## 7.5 Counterexample, repair and re-verification lineage

The committed example performs this loop:

```text
original policy artifact
        ↓
passing verification artifact

original policy
        ↓
seeded policy defect
        ↓
unsafe policy artifact
        ↓
failed verification artifact
        ↓
exact counterexample row and action cell
        ↓
reviewed repair
        ↓
repaired policy artifact with new identity
        ↓
passing re-verification artifact
```

Automatic repair is not implemented. The project demonstrates an identity-bearing, reviewable repair and promotion path.

## 7.6 Criticality-first inspection

The Q-bearing source can produce a deterministic view that places high-criticality states first.

This establishes that a principled consequence metric can drive spatial inspection order. It does not yet establish that people or automated reviewers detect faults faster using this view.

---

# 8. Wider Capability Surface

ZeroModel is larger than the policy reader.

| Capability | Purpose |
|---|---|
| Immutable artifact kernel | Deterministic identity, mapping and provenance |
| Policy lookup | State-addressed action selection and cell trace |
| Policy diagnostics | Criticality and decision margin |
| Policy properties | Exhaustive finite checks and verification artifacts |
| View profiles | Multiple deterministic views over one source |
| Spatial optimization | Optimize an explicit top-left mass objective |
| Decision manifolds | Track spatial changes across scored panel sequences |
| PHOS packing | Sort-pack and top-left concentration |
| Composition | Numeric fuzzy AND/OR/NOT/XOR/add/subtract |
| Comparison | Baseline-versus-target field differences |
| Bundle serialization | Lossless `.vpm` round-trip |
| Rendering | PNG and SVG views |
| Hierarchy | Reduced multi-level fields |
| Edge gates | Small top-left decision consumers |
| Controller | EDIT/RESAMPLE/ESCALATE/STOP/SPINOFF signals |
| Learning traces | Before/after/held-out/regression artifacts |
| Training progress | Checkpoint telemetry and warnings |
| Tracker adapters | Generic, TensorBoard-style, W&B-style and Trackio-style exports |
| Critic artifacts | Risk-first maps from critic/evidence/policy scores |

The unifying idea is:

```text
scored information
        ↓
deterministic identity-bearing artifact
        ↓
different consumers
```

The strength of evidence is not equal across all modules. The claims audit separates validated capabilities, implemented capabilities with thin evidence, unvalidated claims and claims requiring reframing.

---

# 9. What ZeroModel Is Built On

ZeroModel does not claim that each ingredient is historically new.

## 9.1 Tabular policies and Q-tables

State-action lookup, tabular value functions and deterministic argmax are established. ZeroModel does not claim novelty for lookup.

Its proposed contribution is the reusable artifact contract around a scored policy surface: canonical identity, explicit layout, source-to-view mapping, rendering, serialization, candidate retention, artifact-linked decision traces, verification and repair lineage.

## 9.2 Decision tables and DMN

Business decision tables already map conditions to actions, and Decision Model and Notation formalizes executable decisions. A decision table could become a ZeroModel policy producer.

ZeroModel’s additional emphasis is on scored alternatives, visual spatial organization, content identity, multiple views and exact artifact/cell traces.

## 9.3 Partial evaluation and program specialization

Partial evaluation moves work upstream by specializing a general program against known inputs and leaving a residual runtime program.

ZeroModel is not a general partial evaluator. The shared idea is staging: work that no longer depends on runtime novelty can be performed before runtime.

## 9.4 Materialized views, caching and precomputation

Databases, compilers and distributed systems routinely materialize stable computation. ZeroModel asks whether bounded portions of intelligent policy can be materialized while preserving their evidence surface and governance metadata.

## 9.5 Content-addressed artifacts

Content-addressed systems establish identity from canonical contents. ZeroModel applies this discipline to scored policy and evidence artifacts.

Artifact identity is foundational for versioning and lineage, but signatures and trusted attestations remain separate concerns.

## 9.6 Visual analytics and matrix views

Heatmaps and visual inspection of tables are established. ZeroModel does not claim novelty for rendering a matrix.

The relevant claim is that spatial layout is explicit and deterministic, preserved inside the artifact and mapped back to source coordinates.

## 9.7 VIPER

VIPER — *Verifiable Reinforcement Learning via Policy Extraction* — is the closest precedent on the policy-compilation axis.

VIPER uses a neural policy and Q-function as an upstream oracle, then extracts a smaller decision-tree policy that can be analyzed more tractably.

The shared architecture is:

```text
powerful policy producer upstream
        ↓
smaller structured policy downstream
```

VIPER’s Q-DAGGER method also shows that the complete action-value structure carries important signal through its use of criticality.

The distinction is complementary:

```text
VIPER:
extract a tractable policy
so behavioural properties can be verified

ZeroModel:
place a durable artifact contract around scored policy,
candidate evidence, identity, layout, traces,
verification results and repair lineage
```

A VIPER tree could become a ZeroModel producer.

## 9.8 External memory, situated systems and stigmergic intuition

ZeroModel externalizes reusable policy into an environmental object addressed by later consumers. This resembles external memory, routing tables, situated action, environmental coordination and stigmergic systems.

The current artifact is not fully stigmergic because runtime agents do not necessarily modify it. A stronger version would allow controlled evidence accumulation, reviewed policy updates and promotion into future artifact versions.

## 9.9 Sigstore and in-toto as complementary layers

ZeroModel identity answers:

> Which exact artifact produced the decision?

Signing and supply-chain systems answer who produced it, who approved it, which build generated it and whether it passed an authorized promotion process.

These are complementary rather than competing systems.

---

# 10. Strongest Current Claims

The project can currently make these defensible claims.

## Artifact contract

ZeroModel can turn a scored table into a deterministic VPM carrying stable identifiers, explicit spatial organization, source mapping, deterministic identity, provenance, renderable fields and lossless bundle serialization.

## Closed finite policy execution

A bounded, enumerable policy can be compiled into an addressable VPM and consumed without invoking a model or source-policy function at decision time.

## Exact decision trace

A lookup can return the selected action, complete candidate vector and exact artifact/source/view cell that produced it.

## Q-bearing evidence

Criticality and decision margin can be preserved as evidence while remaining excluded from action selection.

## Exhaustive finite properties

Named declarative row-level properties can be checked across every source row.

## Linked verification artifacts

Verification results can become deterministic artifacts linked to the exact checked policy and exact property specification.

## Counterexample lineage

A defect, failed verification result, exact counterexample, reviewed repair, new policy identity and passing re-verification can be retained as artifact lineage.

## Deterministic views and spatial/temporal consumers

The same source can produce deterministic task-specific views, an explicitly optimized spatial profile and temporal geometric summaries.

## Evidence artifacts

Training telemetry, learning observations and critic outputs can become deterministic artifacts for inspection.

---

# 11. Claims Not Yet Established

Do not treat these as solved:

- general formal verification of continuous dynamics;
- temporal safety or liveness proofs;
- universal policy correctness;
- safety certification;
- automatic property discovery;
- automatic policy repair;
- open-world generalization;
- reliable approximate addressing for large continuous spaces;
- criticality-weighted quantization benefits;
- improved human inspection performance;
- planet-scale traversal;
- infinite memory;
- millisecond performance on tiny hardware;
- a 25 KB memory footprint;
- self-describing executable PNGs;
- survival of arbitrary lossy image transformations;
- automatic discovery of the semantically best view;
- real-world hallucination detection;
- explanation of why a model produced its original scores;
- trusted authorship or approval from a content hash alone;
- general semantic equivalence between artifacts;
- automatic edge/cloud symmetry;
- a complete production viewer with pointer navigation.

---

# 12. Likely Deployment Regimes

## Regime One: Fully compiled finite policy

The domain is closed and enumerable. The complete policy can be built, tested and deployed. This is the current shooter regime.

## Regime Two: Compiled common path with intelligent fallback

Known, repeated and reviewed cases use a compiled artifact. Novel, uncertain or out-of-domain cases return to a model, planner, search process, expert or human reviewer.

A reviewed resolution may later become part of a new artifact. This is likely the most practical near-term architecture.

## Regime Three: Hierarchical or approximate policy

The state space cannot be enumerated directly. The system requires abstraction, quantization, hierarchy, approximate addressing, sub-artifacts, uncertainty and fallback policies.

Criticality may determine where to allocate finer resolution and stronger checking, but this is a research hypothesis rather than a validated result.

---

# 13. Current Development and Research Process

## 13.1 Claim-first discipline

Public claims have stable claim IDs and are classified as Verified, Implemented, Research or Vision. The claims audit maps public wording to repository evidence and required next proof.

## 13.2 Implementation before promotion

A capability is not validated merely because code exists. Promotion requires tests, fixtures, reproducible outputs, explicit boundaries, documentation and claims-audit updates.

## 13.3 Pull-request workflow

Features are developed through focused branches and pull requests. CI checks supported Python versions, package tests, source distribution, wheel build and metadata.

## 13.4 Reproducibility artifacts

Important examples should produce `.vpm` bundles, JSON results, rendered views, stable artifact IDs, exact counterexamples and persisted replay evidence.

## 13.5 Adversarial external review

External AIs are asked to find prior art, challenge novelty, identify overclaims, propose stronger experiments and inspect implementation choices.

Useful findings are implemented. Criticism that merely reduces the project to one familiar component is adjudicated rather than accepted automatically.

## 13.6 Publication discipline

Conceptual articles should describe ZeroModel as a durable architecture rather than a sequence of release-specific additions. Version numbers belong mainly in reproducibility and release records.

---

# 14. Proposed Next Stages for the Project

The roadmap below is ordered by likely information gain.

## Stage A — Release and baseline hygiene

### A1. Complete the production release path

- complete TestPyPI clean-environment verification;
- publish the stable package;
- tag the tested commit;
- preserve generated example outputs;
- ensure README, package metadata and claims audit agree.

### A2. Baseline comparison

Implement the same bounded policy using:

- plain Python dictionary or NumPy table;
- conventional decision tree;
- ZeroModel VPM.

Compare:

- behaviour fidelity;
- serialization;
- identity;
- candidate retention;
- trace detail;
- mutation detection;
- policy diffing;
- verification linkage;
- artifact size;
- lookup latency.

Do not frame this as proving VPM lookup is faster than a dictionary. The goal is to isolate the value and cost of the artifact contract.

### A3. Non-game fixture

Build one serious but bounded fixture, preferably:

```text
software deployment evidence
        ↓
DEPLOY / CANARY / HOLD / ROLL_BACK
```

or:

```text
security evidence
        ↓
ALLOW / CHALLENGE / REVIEW / QUARANTINE
```

It should include meaningful alternatives, out-of-domain handling, named properties, one seeded counterexample, repair lineage and comparison with a table or rule implementation.

This is the highest-value step for showing generality beyond the arcade world.

---

## Stage B — Test the visual thesis

### B1. Human inspection study

Compare:

- raw table;
- decision log;
- source-order VPM;
- criticality-first VPM.

Seed policy defects and measure detection time, accuracy, false positives, reviewer confidence and ability to identify the exact source cell.

### B2. Automated inspection benchmark

Give the same artifacts to automated review consumers and compare whether criticality-first or risk-first ordering reduces search steps, tokens, rows inspected, time to counterexample and missed defects.

### B3. Viewer

Build a small static or local viewer that supports:

- click a cell;
- inspect source value;
- inspect candidate row;
- inspect provenance;
- inspect parents;
- compare two artifacts;
- show property results;
- follow counterexample-to-repair lineage.

The viewer should consume real `.vpm` artifacts rather than a separate demo format.

---

## Stage C — Extend verification carefully

### C1. Dynamic property checker

Add transition-system properties to the finite fixture.

Examples:

- a target is eventually cleared;
- cooldown prevents immediate repeated fire;
- no trajectory enters a forbidden state;
- every wave terminates within a bound.

Compare exhaustive transition scans with an SMT-backed encoding and, where useful, a decision-tree representation.

### C2. Property-set identity and governance

Treat policy, property set, checker, verification result and approval decision as distinct governed objects.

Explore Sigstore or in-toto integration for trusted promotion.

### C3. Counterexample minimization

When a property fails, produce the smallest failing row, shortest failing trajectory, minimal changed-cell set and behavioural diff against the prior artifact.

Do not jump directly to autonomous repair. Improve diagnosis first.

---

## Stage D — Criticality-weighted representation

Under a fixed artifact budget, compare uniform discretization with criticality-weighted discretization.

Measure:

- action fidelity;
- value error;
- critical-state error;
- episode return;
- failure rate;
- artifact size;
- out-of-domain fallback rate.

This tests whether VIPER’s critical-state intuition transfers to approximate VPM representation. A negative result would still be valuable.

---

## Stage E — Hybrid compilation and policy promotion

Build a controlled Regime Two prototype:

```text
known state
    → compiled artifact

unknown or low-confidence state
    → model or human fallback
    → decision evidence
    → review
    → regression suite
    → candidate new artifact
    → verification
    → promotion
```

Research questions:

- What qualifies a case for compilation?
- How is uncertainty represented?
- What evidence is required for promotion?
- How are regressions checked?
- How is rollback performed?
- How are old decisions linked to the policy version active at the time?
- When should a compiled region be invalidated?

This may be the most important long-term systems contribution.

---

## Stage F — Edge and portability

Build a minimal reader in Rust, C or MicroPython using the same committed artifact bytes as the Python implementation.

Measure startup time, lookup latency distributions, memory use, artifact size and trace completeness.

Only after this should the project make edge-performance claims.

---

## Stage G — Hierarchical policy artifacts

Develop stable child-artifact IDs, a resolver interface, parent/child tiles, a traversal contract, bounded lookup paths, out-of-domain fallback and hierarchy benchmarks.

This is a later architecture, not a prerequisite for validating the current finite artifact contract.

---

## Stage H — Real evidence integrations

Add sanitized fixtures from actual systems:

- TensorBoard;
- Weights & Biases;
- Trackio;
- RAG or critic evaluation;
- software deployment;
- anomaly triage.

The aim is not to add every integration. It is to demonstrate that the artifact contract survives real, imperfect evidence.

---

# 15. Proposed Next Stages for the Development Process

## Process 1: Maintain one canonical project map

Create one document mapping:

```text
research question
→ claim IDs
→ implementation modules
→ fixtures
→ benchmarks
→ publications
→ open issues
```

The claims register, claims audit and backlog should not drift into independent systems.

## Process 2: Use experiment packets

Every research claim should have a compact experiment packet containing:

- hypothesis;
- baseline;
- fixture;
- metric;
- falsification condition;
- expected artifacts;
- hardware/environment;
- claim IDs affected;
- decision rule for promotion.

## Process 3: Separate product and research tracks

**Product track:** stable artifact API, package quality, viewer, adapters, portability and deployment workflow.

**Research track:** human inspection, criticality allocation, dynamic verification, hybrid promotion and hierarchical policy.

The tracks should share the artifact contract but have different acceptance standards.

## Process 4: Require baseline-first development

Before creating a new ZeroModel mechanism, implement the simplest credible baseline: dictionary, NumPy table, decision tree, SQLite table, DMN rule table or ordinary JSON artifact.

This clarifies whether each new feature creates enough value to justify complexity.

## Process 5: Track negative results

A failed experiment should produce a committed report rather than disappear.

Negative results are especially important for visual inspection, criticality-weighted layout, PHOS concentration, temporal manifolds and edge performance.

## Process 6: Invite independent reproduction

Prepare a small artifact-evaluation package another developer can run without project context.

It should include a clean install, one command, expected artifact IDs, expected counterexample, expected repaired verification result and baseline comparison.

## Process 7: Reduce version language in conceptual publications

Describe persistent architecture as ZeroModel. Record versions and commits in reproducibility appendices.

---

# 16. Questions for External Reviewers

## A. Concept and novelty

1. What is the strongest accurate description of ZeroModel?
2. What is genuinely distinctive?
3. Which parts are ordinary engineering?
4. Is “compiled policy artifact” the right abstraction?
5. Is “Visual Policy Map” technically justified, or is another name better?
6. Does the combination of identity, layout, candidate retention, traces and verification lineage create a meaningful systems contribution?
7. What existing work most directly overlaps with the complete system?

## B. Architecture

1. Is `ScoreTable → LayoutRecipe → VPMArtifact → Consumer` a coherent minimal kernel?
2. Which fields belong in canonical identity?
3. Should provenance affect artifact identity?
4. Should different layouts produce different artifact IDs, view IDs, or both?
5. Is the distinction between action and evidence columns sound?
6. Is the verification artifact correctly modeled as another VPM, or should it be a separate artifact type?
7. What is missing from the lineage model?
8. Where is the architecture unnecessarily coupled to two-dimensional layout?

## C. Verification

1. Is exhaustive row-level checking useful enough to justify the word verification when bounded carefully?
2. What property language should be supported next?
3. Should dynamic properties use explicit transition systems, temporal logic, SMT or model checking?
4. How should property sets be versioned and approved?
5. How should failed checks represent minimal counterexamples?
6. What would be needed before using this in safety-sensitive systems?

## D. Criticality

1. Is best-minus-worst the right evidence metric to preserve?
2. Which other metrics should be first-class?
3. Should advantage, entropy, regret, uncertainty or calibration be included?
4. Does criticality-first layout have a plausible inspection benefit?
5. What experiment would falsify the criticality-weighted representation hypothesis?
6. Does VIPER provide enough theoretical motivation, or is the transfer too weak?

## E. Product and use cases

1. Which practical domain is the best first non-game fixture?
2. Where would a VPM add more value than a normal table, policy engine or event log?
3. Which user would adopt this first?
4. Is the likely product a library, artifact format, policy registry, verification tool, viewer, deployment gateway or research framework?
5. What should be removed from the package to make the project easier to understand?

## F. Research programme

1. Which three experiments provide the highest information gain?
2. Which current research directions are distractions?
3. What would make the work publishable as a systems or ML paper?
4. What are the strongest baselines?
5. What independent reproduction would be persuasive?
6. Which claims should be narrowed, split or deleted?

## G. Process

1. Is the claim-registry process useful or too heavy?
2. How should claim IDs, audits, experiments and issues be unified?
3. What release cadence makes sense?
4. Should research prototypes live in the core package or separate experimental modules?
5. How can external AI review be used without producing design-by-consensus?
6. What evidence should be required before promoting a Research claim to Verified?

---

# 17. Required Response Format

## 1. Executive verdict

In no more than 500 words, state:

- project quality;
- strongest contribution;
- largest weakness;
- whether the project is worth continuing;
- recommended primary direction.

## 2. Your model of ZeroModel

Explain the project in your own words. This tests whether the current explanation is coherent.

## 3. Prior-art map

Use this table:

| ZeroModel element | Closest prior work | Same | Different | Implication |
|---|---|---|---|---|

Do not state that no prior work exists unless you performed a credible search.

## 4. Architecture review

Classify findings as:

- defect;
- design risk;
- missing feature;
- unnecessary complexity;
- good decision.

Rank each by severity.

## 5. Claims review

Classify important claims as:

- supported;
- plausible but untested;
- overstated;
- false or confused;
- unclear.

## 6. Experiment roadmap

Recommend the five highest-value experiments. For each include:

- hypothesis;
- baseline;
- dataset or fixture;
- metric;
- falsification condition;
- estimated effort;
- claim IDs affected.

## 7. Product roadmap

Recommend:

- first user;
- first serious domain;
- minimum useful product;
- features to defer;
- possible commercial or open-source positioning.

## 8. Process roadmap

Recommend changes to issue design, PR structure, claims governance, release process, external review and reproducibility.

## 9. Kill criteria

State what experimental results would indicate that:

- the visual representation is not useful;
- the artifact contract is not worth its complexity;
- criticality-guided layout should be abandoned;
- the project should narrow to a smaller library;
- the wider research programme should stop.

## 10. Final prioritized plan

Provide:

- next 30 days;
- next 90 days;
- next 6 months.

Use explicit priorities:

- P0;
- P1;
- P2;
- defer;
- reject.

---

# 18. Review Boundaries

Do not criticize ZeroModel merely because lookup, tables, hashing, rendering or verification each existed previously. The relevant question is whether the complete artifact contract is coherent and useful.

Do not praise ZeroModel merely because it combines many capabilities. The relevant question is whether those capabilities belong together and produce measurable value.

Do not assume a rendered image is the artifact.

Do not assume the content hash establishes trust.

Do not treat finite row checking as universal formal verification.

Do not treat a successful toy fixture as open-world evidence.

Do not treat the number of registered claims as the number of historical inventions.

Do not assume every module has equal evidence.

Do not recommend autonomous repair before diagnosis, property coverage and promotion governance are sound.

The desired review is neither defensive nor dismissive.

It should identify the smallest, strongest and most defensible version of ZeroModel—and the experiments that could justify expanding it.
