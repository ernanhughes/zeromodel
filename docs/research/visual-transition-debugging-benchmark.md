# Visual Transition Debugging Benchmark

- Implementation and reproduction: [`examples/visual_transition_benchmark/README.md`](../../examples/visual_transition_benchmark/README.md)
- Frozen result record: [`docs/results/visual-transition-debugging-v1/`](../results/visual-transition-debugging-v1/)
- Claims boundary: [`docs/claims-audit.md`](../claims-audit.md)

## Executive finding

ZeroModel demonstrates a measurable but narrow visual-transition debugging capability in the deterministic `TinyArcadeShooter` environment.

On the frozen evaluation split, the ZeroModel transition analyzer achieved perfect **visible changed-component attribution** under the benchmark's declared component mapping and outperformed the committed raw pixel-difference baseline on that metric. More importantly, it detected selected visual contract violations that raw pixel differencing cannot express directly, including:

- a component declared stable changing unexpectedly;
- a component declared required-to-change remaining unchanged;
- a visually small mutation that fell below the raw pixel baseline's configured minimum region size.

The benchmark also exposed clear structural limits.

The current representation does not reliably determine:

- whether a component changed in the correct direction;
- whether it changed to the correct value;
- whether the correct target changed;
- whether a hidden or suppressed event occurred when the supplied frames contain insufficient evidence.

The result therefore supports a bounded claim:

> In a deterministic visual domain with a known component schema, ZeroModel can compile selected visual transition expectations into inspectable spatial contracts and use them to localize visible changed components and selected missing or unexpected changes.

It does not establish general visual perception, semantic understanding from pixels, causal diagnosis, open-world generalization, or production readiness.

---

## Why this benchmark was created

The ZeroModel perception work had accumulated a substantial architecture around:

- observation encoding;
- transition evidence;
- conformance;
- discovery;
- candidate validation;
- promotion;
- materialization;
- activation;
- rollback;
- certification;
- governance.

That architecture contained many carefully designed invariants, but continued stage-by-stage expansion was no longer producing proportionate empirical understanding.

The benchmark was created to stop architectural extension and answer one falsifiable question:

> Does the existing ZeroModel visual representation provide useful transition-debugging evidence beyond straightforward pixel differencing?

The benchmark deliberately avoided introducing another perception stage. It reused only the smallest existing ZeroModel components needed to analyze a controlled visual transition and compared them with simple baselines.

The objective was not to prove ZeroModel correct. The objective was to determine whether a useful visual-testing instrument existed inside the larger architecture.

---

## Research question

The primary research question was:

> Can ZeroModel identify which visual components changed, or violated an expected transition contract, more accurately or more usefully than a straightforward raw pixel-difference baseline?

The benchmark evaluated two related capabilities.

### Visible change attribution

Given a frame before and after an action:

- which declared components changed;
- which changed components did the analyzer identify;
- which unchanged components did it implicate incorrectly?

### Transition contract conformance

Given an action and a declared expectation:

- did a component that should remain stable change;
- did a component that should change remain unchanged;
- did an unrelated component change unexpectedly?

These are distinct from value-level correctness.

The benchmark did not initially attempt to determine:

- whether movement was in the correct direction;
- whether a numeric state changed by the correct amount;
- whether the correct target was affected;
- whether an event occurred internally but left no sufficient visual trace.

---

## Repository location

The implementation is located at:

```text
examples/visual_transition_benchmark/
```

The benchmark includes:

```text
dataset.py
baselines.py
zeromodel_adapter.py
discovery_demo.py
metrics.py
render.py
report.py
run.py
README.md
tests/
```

The implementation remains domain-specific and imports the perception package.

No arcade-specific logic was added to the domain-independent perception package.

No existing file under `zeromodel/perception` was modified for the benchmark.

---

## Environment and evaluated revision

The frozen run reported:

| Item | Value |
|---|---|
| Repository revision | `80b608d` |
| Python | `3.11.4` |
| NumPy | `2.2.3` |
| Development transitions | `720` |
| Evaluation transitions | `2,160` |
| Ordinary categories | `8` |
| Fault categories | `10` |
| Benchmark tests | `24 passed` |
| Existing perception tests | `212 passed` |
| Full benchmark runtime | approximately `62 seconds` |

The principal benchmark command was:

```bash
PYTHONPATH=examples python -m visual_transition_benchmark.run \
  --dev-episodes 40 \
  --eval-episodes 120 \
  --output-dir artifacts/visual_transition_benchmark
```

The benchmark test suite was executed with:

```bash
PYTHONPATH=examples python -m pytest \
  examples/visual_transition_benchmark/tests -q
```

The existing perception regression suite was executed with:

```bash
python -m pytest packages/perception/tests -q
```

The benchmark completed without crashes or perception regressions.

---

## Domain and observation model

The benchmark uses the repository's deterministic `TinyArcadeShooter` domain and its existing frame renderer.

Each transition record contains sufficient information to reproduce and evaluate one state change:

```text
transition_id
episode_id
step_number
seed
action
frame_before
frame_after
state_before
state_after
component annotations
expected changed components
observed changed components
fault type
fault flag
```

The benchmark separates deployable visual evidence from privileged evaluation evidence.

The ZeroModel adapter receives only the information available to the tested analyzer:

- `frame_before`;
- `frame_after`;
- action;
- non-semantic transition metadata such as transition identity and step number.

Privileged component annotations and ground-truth state are used only by:

- the dataset generator;
- the privileged baseline;
- metric calculation;
- diagnostic rendering.

They are not leaked into the ZeroModel adapter.

---

## Dataset design

The dataset contains deterministic ordinary transitions and deliberately injected faults.

The generator records all relevant seeds and transition identities, allowing exact reproduction.

### Ordinary transition categories

The committed evaluation contains eight ordinary categories representing valid behavior in the arcade domain.

These cover ordinary combinations of:

- movement;
- firing;
- tank stability;
- alien stability or removal;
- cooldown state changes;
- background stability;
- transitions where no relevant visual component should change.

The precise category names are defined in the benchmark implementation and emitted in transition-level results.

### Fault categories

The committed evaluation contains ten deterministic injected-fault categories.

The reported examples include faults such as:

- FIRE selected but no projectile or hit effect is produced;
- FIRE selected but cooldown does not activate;
- an alien changes unexpectedly;
- the tank moves when the selected action should not move it;
- the tank moves in the wrong direction;
- cooldown changes in an invalid context;
- the background changes unexpectedly;
- an expected target remains unchanged;
- an unrelated component changes alongside the expected transition.

Every injected fault has a known ground-truth modification and expected affected-component set.

The benchmark does not use uncontrolled random corruption as a substitute for fault semantics.

---

## Split design

The generated dataset is divided into:

```text
development split
evaluation split
```

Splits are owned by complete episodes rather than individual frames.

No episode contributes transitions to both splits.

The benchmark verifies split disjointness at runtime.

The development split is used for implementation checks and any fixed parameter selection required by the benchmark.

The evaluation split is held separate for final metric calculation.

The frozen run contains:

| Split | Episodes | Transitions |
|---|---:|---:|
| Development | `40` | `720` |
| Evaluation | `120` | `2,160` |

Each of the eighteen transition categories contributes `120` evaluation transitions.

---

## Systems compared

The benchmark compares three systems on the same transition records.

### Raw pixel-difference baseline

The raw baseline computes visible differences between `frame_before` and `frame_after`.

It then groups sufficiently large changed regions according to its declared threshold and minimum-component configuration.

The baseline answers:

> What visibly changed between the supplied frames?

It does not encode:

- what should have changed;
- what should have remained stable;
- which absent change constitutes a failure;
- the semantic meaning of a changed region.

One benchmark finding depends on this distinction: a one-pixel background mutation was discarded by the raw baseline's configured `min_component_size=2`, while the ZeroModel field-level evidence detected the mean change within the declared background field.

This is a valid result for the committed baseline configuration, not proof that every possible pixel-difference implementation must miss the mutation.

### Privileged component baseline

The privileged baseline uses full ground-truth component annotations and state transitions.

It answers:

> Which declared environment components actually changed or violated the expected transition?

This baseline is not deployable.

It acts as an upper reference for:

- visible changed-component attribution;
- missing expected changes;
- unexpected component changes;
- fault localization.

### ZeroModel transition analyzer

The ZeroModel adapter uses the narrowest relevant subset of the current perception implementation:

- P4A field partitioning;
- P18A transition evidence;
- P18B transition conformance;
- P18C recurrent unexplained-transition discovery as a secondary cohort demonstration.

The core analyzer produces structured evidence about:

- changed fields;
- mapped components;
- components expected to change;
- components expected to remain stable;
- unexpected changed components;
- required components that did not change;
- conformance or unexplained-transition status.

The benchmark deliberately bypasses:

- candidate promotion;
- materialization;
- activation;
- rollback;
- certification;
- governance execution;
- operational health.

Those components were not necessary to test the visual-transition research claim.

---

## Component model

The benchmark maps visual fields to a bounded set of declared components, including:

```text
tank
alien
projectile or firing effect
cooldown or status region
background
```

The precise mapping is deterministic and belongs to the controlled domain.

The ZeroModel result should therefore be understood as:

> field-level visual evidence mapped through a known component schema.

It is not autonomous object discovery.

It is not semantic labeling learned from raw pixels.

It is not open-world recognition.

---

## Metrics

The benchmark reports metrics separately for ordinary, faulty, aggregate, and per-category transitions where applicable.

### Visible changed-component attribution

This evaluates whether a system identifies the declared components that visibly changed.

The principal measures are:

- micro precision;
- micro recall;
- micro F1;
- exact changed-component-set accuracy.

The headline `1.000` ZeroModel result refers to this bounded metric.

It must not be described simply as "perfect component fault attribution."

### Missing expected-change detection

This evaluates whether the system notices that a component required to change remained unchanged.

Examples include:

- cooldown should activate after FIRE but does not;
- tank movement is expected but absent;
- a declared target should change but remains stable.

Raw pixel differencing cannot represent this condition by itself because the failure is the absence of a visual difference.

### Unexpected-change detection

This evaluates whether a system identifies a component that changed despite being declared stable under the transition contract.

Examples include:

- tank movement during a non-movement action;
- a background mutation;
- an unrelated component changing.

### False implicated components

This counts declared components marked as relevant even though they neither changed nor violated an expectation.

This prevents a high-recall system from appearing useful by marking every component.

### Relative usefulness against pixel difference

The benchmark classifies each transition as:

- ZeroModel better;
- equal;
- raw pixel difference better;

under the benchmark's declared comparison rule.

A ZeroModel result is more useful when it provides correct component-level or contract-level localization with fewer false implications, or detects a missing expected change that raw pixel difference cannot express.

---

## Aggregate results

The frozen evaluation produced the following headline metrics:

| Metric | Raw pixel difference | Privileged baseline | ZeroModel |
|---|---:|---:|---:|
| Visible changed-component attribution micro-F1 | `0.937` | `1.000` | `1.000` |
| Missing expected-change detection | not applicable by construction | `1.000` | `0.681` |
| Unexpected-change detection | not applicable by construction | `1.000` | `0.500` |
| False-alarm rate on correct transitions | not reported as a contract system | `0.000` | `0.125` |
| Better than pixel difference | — | — | `23.0%` |
| Equal to pixel difference | — | — | `77.0%` |
| Worse than pixel difference | — | — | `0.0%` |

These figures must be interpreted together.

The benchmark supports a positive finding, but the positive result is narrow.

ZeroModel did not become a complete fault-diagnosis system.

Its primary advantage is that it converts visual change into component-level evidence and evaluates selected declarative transition expectations.

---

## Where ZeroModel helped

### Declared-stable background mutation

Representative artifact:

```text
eval-0000-0015.png
```

A one-pixel background mutation was injected.

The raw pixel-difference baseline's minimum connected-component size removed the mutation from its derived region output.

ZeroModel's field-level comparison detected the change within the declared background field and identified a stable-component violation.

This demonstrates the value of testing change against declared spatial fields rather than treating each raw connected component as independently meaningful.

It does not demonstrate universal superiority over all pixel-difference configurations.

### Missing cooldown activation

Representative artifact:

```text
eval-0000-0009.png
```

The FIRE action required a cooldown-region change, but the faulty transition suppressed that change.

Raw pixel differencing had no changed pixels to classify in the absent cooldown transition.

ZeroModel compared the observed transition with the declared must-change expectation and reported the missing change.

This is one of the benchmark's strongest positive findings.

It demonstrates that a visual transition contract can represent failure by absence, rather than only by unexpected presence.

### Tank movement during the wrong action

Representative artifact:

```text
eval-0000-0011.png
```

The tank changed during an action under which it was declared stable.

ZeroModel identified the tank as an unexpected changed component.

The result is more useful than a raw changed-pixel mask because it binds the visible mutation to a declared component and expectation.

---

## Where ZeroModel failed

### Suppressed firing event

Representative artifact:

```text
eval-0000-0008.png
```

For `fire_no_projectile`, the analyzer reported the transition as conformant.

The supplied frame pair and action did not provide sufficient information for the current contract to distinguish:

- a legitimate miss;
- a suppressed projectile;
- an invalid target;
- an internal engine failure;
- another hidden event.

This is an observability problem as well as a representation problem.

A stronger diagnosis may require:

- a visible projectile phase;
- additional temporal frames;
- an explicit expected target;
- richer observation state;
- a contract over intermediate events.

The benchmark therefore does not support a claim that ZeroModel detects hidden events from frame pairs.

### Alien and cooldown anomalies outside hard contracts

Representative artifacts:

```text
eval-0000-0010.png
eval-0000-0014.png
```

Some alien and cooldown anomalies outside the strongest action-specific expectations reached only a softer unexplained-transition category.

The system did not reliably promote them to hard contract violations.

This indicates that:

- the current expectation vocabulary is incomplete;
- not every anomalous component relation is encoded;
- unexplained evidence should not be confused with a proven fault.

### Wrong-direction movement

Representative artifact:

```text
eval-0000-0012.png
```

The current component-level representation can determine that the tank changed.

It cannot reliably determine whether the tank moved left when it should have moved right, or vice versa.

Both outcomes map to the same changed component:

```text
tank changed
```

Direction is a value or relation over the component, not a component-presence property.

The current scoring granularity therefore makes some wrong-direction faults invisible.

The reported `0.681` missing-change detection figure is also affected by a boundary-clipping coincidence in the wrong-direction category.

At a movement boundary, a wrong-direction command may produce no movement, making the transition appear like a missing expected change.

This can inflate missing-change detection without demonstrating true direction understanding.

Future runs should:

- generate direction tests away from movement boundaries;
- classify boundary-limited transitions separately;
- report value-level direction accuracy independently from missing-change detection.

---

## Three capability levels revealed by the benchmark

### Presence-level transition reasoning

Demonstrated within the controlled benchmark:

```text
Did this declared component change?
Did a component declared stable change?
Did a component required to change remain unchanged?
```

This is the strongest current result.

### Value-level transition reasoning

Not demonstrated:

```text
Did the tank move in the correct direction?
Did cooldown change to the correct value?
Did the expected target change?
Did a field increase, decrease, or reach the required state?
```

The current representation largely treats components as changed or unchanged.

It does not yet compile enough typed component value semantics.

### Hidden-event reasoning

Not demonstrated and sometimes unidentifiable from current observations:

```text
Was a projectile suppressed internally?
Was a shot blocked?
Was the intended target invalid?
Did an internal transition occur without a visible trace?
```

This capability may require richer observations rather than a more complex analysis layer.

---

## Secondary recurrent-discovery demonstration

The benchmark also exercised P18C recurrent unexplained-transition discovery as a secondary cohort demonstration.

This part of the experiment shows that repeated unexplained transition patterns can be grouped and surfaced for review.

It does not establish:

- semantic ontology discovery;
- causal attribution;
- autonomous component naming;
- automatic fault explanation.

Its value is narrower:

> recurrent unexplained visual-transition evidence can be collected into inspectable cohorts.

This may be useful for identifying repeated failure signatures after basic conformance analysis.

---

## Visual artifacts

The full run generated diagnostic panels for:

- all faulty evaluation transitions;
- representative ordinary transitions.

The reported run produced `1,216` PNG diagnostic panels.

Each panel includes the relevant combination of:

- frame before;
- frame after;
- raw pixel difference;
- ground-truth changed regions;
- ZeroModel predicted regions;
- expected components;
- observed components;
- missing components;
- unexpected components;
- action;
- category;
- fault identity.

The generated HTML index supports inspection of:

- failures;
- ZeroModel-only successes;
- pixel-baseline-only successes;
- false positives;
- false negatives.

Large generated artifact collections should normally remain reproducible output rather than all being committed to Git.

A compact frozen evidence package should retain:

- aggregate JSON results;
- environment metadata;
- benchmark summary;
- representative successes;
- representative failures.

---

## Test coverage

The benchmark adds twenty-four focused tests.

### Dataset tests

The tests verify that:

- the same seed produces identical transitions;
- development and evaluation episodes remain disjoint;
- frame and state annotations agree;
- injected faults alter only their declared targets;
- ordinary transitions contain no injected fault.

### Metric tests

The tests verify that:

- perfect predictions produce perfect metrics;
- empty predictions have zero recall;
- predicting all components harms precision;
- missing expected changes are counted correctly;
- unexpected extra changes are counted correctly.

### Baseline tests

The tests verify that:

- raw pixel difference detects a visible mutation;
- raw pixel difference alone cannot represent an absent expected transition;
- the privileged baseline agrees with declared component changes.

### ZeroModel adapter tests

The tests verify that:

- output coordinates align with the original frame;
- field-to-component mapping is deterministic;
- undeclared components are not invented;
- repeated analysis of the same transition produces identical evidence.

### End-to-end test

The end-to-end smoke test:

- generates deterministic ordinary and faulty transitions;
- runs all three systems;
- computes metrics;
- renders at least one diagnostic artifact.

The existing perception suite also remained green with `212` passing tests.

---

## Threats to validity

### Controlled synthetic domain

The benchmark uses a deterministic renderer and known component schema.

This gives strong ground truth but limits external validity.

The result does not establish performance on:

- natural images;
- variable cameras;
- lighting changes;
- occlusion;
- sensor noise;
- new object types;
- unbounded environments.

### Hand-declared component mapping

The mapping from visual fields to semantic components is supplied by the experiment.

The benchmark does not test autonomous semantic component discovery.

### Baseline configuration

The raw pixel baseline uses fixed thresholds and a minimum component size.

A different baseline configuration might improve some cases, particularly tiny mutations.

The benchmark's stronger advantage lies in missing-change and stable-component contracts, which pure pixel difference cannot express without additional expectations.

### Component-level scoring

Component-level scoring can hide value-level errors.

A tank moving in the wrong direction may still count as correctly attributed because the changed component is the tank.

Visible changed-component attribution must therefore remain separate from fault correctness.

### Boundary effects

Movement boundaries can transform a direction error into a no-movement event.

This affects interpretation of the current missing-change detection result.

### Deterministic category balance

The frozen evaluation uses equal counts per category.

Aggregate percentages therefore reflect the benchmark's chosen category distribution, not a measured real-world fault prevalence.

### Single domain

The positive result has not yet been reproduced in another deterministic visual domain.

A second domain is required before making broader visual-contract-testing claims.

---

## What the benchmark demonstrates

Within the tested environment and declared component schema, the benchmark demonstrates that ZeroModel can:

- deterministically partition visual observations into declared fields;
- attribute visible frame changes to known components;
- evaluate selected must-change expectations;
- evaluate selected must-remain-stable expectations;
- identify selected missing and unexpected component changes;
- produce inspectable transition evidence;
- outperform the committed raw pixel-difference baseline on visible changed-component attribution;
- provide more useful localization than the raw baseline on a subset of transitions;
- surface recurrent unexplained transitions for later review.

---

## What the benchmark suggests

The result suggests that ZeroModel's strongest near-term role may be:

> visual contract testing for deterministic or bounded stateful systems.

In this framing, ZeroModel does not attempt to infer all semantics from pixels.

Instead, it compiles known visual structure and expected transitions into inspectable spatial contracts.

This may be useful when:

- the visual layout is stable or bounded;
- components can be declared;
- valid actions imply expected component transitions;
- debugging requires identifying what changed, what failed to change, or what changed unexpectedly.

---

## What the benchmark does not establish

The benchmark does not establish:

- general visual intelligence;
- open-world perception;
- semantic understanding learned from pixels;
- causal discovery;
- causal fault diagnosis;
- correct direction reasoning;
- correct numeric value reasoning;
- correct target reasoning;
- hidden-event detection;
- production readiness;
- robustness to natural visual variation;
- transfer to another environment;
- necessity of the wider governance architecture.

---

## Architecture implications

The benchmark used only a narrow slice of the existing perception architecture:

```text
field partitioning
transition evidence
transition conformance
recurrent unexplained-transition discovery
```

It did not require:

```text
candidate promotion
materialization
activation
rollback
certification
admission
governance execution
operational health
```

This provides an empirical basis for distinguishing two layers.

### Core visual testing instrument

```text
fields
evidence
expectations
conformance
discovery
```

This layer contains the currently demonstrated research value.

### Optional governed lifecycle

```text
promotion
materialization
activation
rollback
certification
governance
```

This layer may become useful when visual contracts are automatically proposed, deployed, revised, or rolled back.

The benchmark did not validate it and did not need it.

Future architecture work should not treat the existence of the second layer as evidence that it is currently necessary.

---

## Supported claim

The strongest claim supported by the frozen result is:

> Within the deterministic TinyArcadeShooter benchmark, ZeroModel's field-based transition evidence and conformance checks attributed visible changes to declared components and detected selected stable-component and required-change violations that raw pixel differencing could not express.

A narrower comparative claim is also supported:

> Under the committed benchmark configuration and component mapping, ZeroModel achieved higher visible changed-component attribution micro-F1 than the committed raw pixel-difference baseline.

Both claims must remain bounded to:

- the deterministic renderer;
- the declared component schema;
- the frozen dataset;
- the recorded baseline configuration;
- the benchmark's declared metrics.

---

## Claims to avoid

Do not claim:

- "perfect component fault attribution";
- "ZeroModel understands whether a component changed correctly";
- "ZeroModel detects hidden events from images";
- "ZeroModel proves why a transition occurred";
- "transition evidence proves causality";
- "ZeroModel always outperforms pixel difference";
- "ZeroModel has general visual perception";
- "ZeroModel discovers semantic objects from pixels";
- "ZeroModel is production ready";
- "the wider governance chain is validated by this benchmark."

The `1.000` score must always be named as:

```text
visible changed-component attribution micro-F1
```

It is not a measure of complete fault diagnosis.

---

## Recommended next experiment

The next experiment should extend the same benchmark from component-presence contracts to value-aware transition contracts.

It should not introduce another stage-oriented architecture.

### Value-aware component records

Add minimal typed values such as:

```text
tank:
    before_x
    after_x
    expected_delta_x

cooldown:
    before_value
    after_value
    expected_value

alien:
    before_alive
    after_alive
    expected_target_id
```

### Value-aware expectations

Compile expectations such as:

```text
MOVE_LEFT:
    tank.delta_x < 0

MOVE_RIGHT:
    tank.delta_x > 0

FIRE:
    cooldown.before == 0
    cooldown.after > 0

PROJECTILE_HIT:
    expected_target.alive changes from true to false
```

### Required comparison

Evaluate:

| System | Evidence |
|---|---|
| Raw pixel difference | Before and after frames |
| Current component-change ZeroModel | Presence-level field evidence |
| Value-aware ZeroModel | Typed field values and expected relations |
| Privileged state baseline | Full environment state |

### Required new metrics

Report separately:

- visible changed-component attribution;
- movement-direction accuracy;
- cooldown-value accuracy;
- target-specific transition accuracy;
- missing-event detection;
- false implicated components;
- boundary-limited transitions.

The research question should be:

> Does compiling typed component values and transition relations improve fault localization beyond detecting whether a component changed?

---

## Reproduction

Implementation and usage instructions are maintained in:

```text
examples/visual_transition_benchmark/README.md
```

The benchmark can be rerun with:

```bash
PYTHONPATH=examples python -m visual_transition_benchmark.run \
  --dev-episodes 40 \
  --eval-episodes 120 \
  --output-dir artifacts/visual_transition_benchmark
```

Expected generated outputs include:

```text
benchmark-results.json
benchmark-summary.md
transition-level-results.jsonl
visual-index.html
artifacts/*.png
```

The complete generated artifact directory is reproducible and may be excluded from Git.

A compact frozen result package should be committed under:

```text
docs/results/visual-transition-debugging-v1/
```

The corresponding bounded public claim should be recorded in:

```text
docs/claims-audit.md
```

---

## Conclusion

The visual-transition debugging benchmark produced the first clear bounded positive result from the newer perception work.

ZeroModel did not demonstrate general visual AI.

It demonstrated something narrower and more defensible:

> known visual structure and expected state transitions can be compiled into inspectable spatial contracts that identify selected missing and unexpected component changes.

The experiment also showed exactly where the current representation fails.

It handles component presence more successfully than component values, target relations, or hidden events.

That boundary provides a concrete research direction.

Future work should now follow an empirical loop:

```text
propose one representational improvement
    ↓
inject faults it should detect
    ↓
compare with simple baselines
    ↓
inspect failures
    ↓
retain or reject the improvement
```

This benchmark should remain the decision instrument for that work.
