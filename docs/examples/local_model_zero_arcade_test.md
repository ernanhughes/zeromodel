# Local Vision Model → ZeroModel Arcade Policy Bridge

## Overview

This example demonstrates how a local vision-language model can operate as an external observation provider for a ZeroModel policy.

The local model observes a rendered arcade frame and reports a small set of visible state variables. ZeroModel validates those observations, converts them into a stable policy-state address, and selects an action from an independently compiled Visual Policy Map.

```text
rendered arcade frame
        ↓
local vision-language model
        ↓
bounded text observation protocol
        ↓
deterministic parsing and validation
        ↓
stable ZeroModel policy row
        ↓
identified VPM policy
        ↓
VPMPolicyLookup
        ↓
LEFT / RIGHT / STAY / FIRE
```

The model performs **perception**.

ZeroModel performs **state validation, policy addressing, action selection, and trace construction**.

The model is never given the action policy and is never asked to choose `LEFT`, `RIGHT`, `STAY`, or `FIRE`.

---

## Why this example exists

ZeroModel can compile a bounded policy into a deterministic artifact and execute that policy without invoking a model at decision time.

That still leaves an important upstream question:

> How does a runtime observation become the stable state address required by the compiled policy?

There are several possible answers:

* direct instrumentation;
* deterministic visual protocols;
* conventional computer vision;
* learned visual models;
* human or system-provided observations.

This example tests one of those paths:

> A local vision-language model observes the frame, while ZeroModel retains authority over the downstream policy decision.

The experiment deliberately avoids turning ZeroModel into a general-purpose perception framework.

The local model remains a replaceable provider outside ZeroModel Core.

---

## Architectural boundary

The central rule of the example is that **perception and policy remain separate**.

### Local model responsibility

The local model receives:

* one rendered frame;
* a description of the visible objects;
* the bounded observation vocabulary;
* the required text response format.

It reports:

* the player tank column;
* whether a target is visible;
* the target column when present;
* whether the weapon is ready or blocked;
* a self-reported confidence value.

### ZeroModel responsibility

ZeroModel:

1. validates the model response;
2. rejects missing, contradictory, malformed, or out-of-range values;
3. constructs a stable policy row identifier;
4. addresses the independently compiled VPM policy;
5. selects the winning action;
6. records the observation, policy identity, action, and evaluation trace.

### What the model does not receive

The local model does not receive:

* the action values;
* the policy table;
* the action-selection rules;
* the expected state;
* the expected action;
* the ground-truth row identifier.

This distinction matters.

For example, the model may report:

```text
TANK_COLUMN: 0
TARGET_COLUMN: 6
COOLDOWN: READY
CONFIDENCE: 95
```

The deterministic adapter converts that observation into:

```text
tank=0|target=6|cooldown=0
```

The compiled ZeroModel policy then returns:

```text
RIGHT
```

The model observed the scene.

ZeroModel selected the action.

---

## Why the provider uses a text protocol

The first local-model experiments attempted to require schema-constrained JSON responses.

That path was unreliable with the tested Ollama and local-model combination. Some requests returned empty visible content even though the model had processed the image.

The working design uses a minimal labelled-text protocol:

```text
TANK_COLUMN: <0-6>
TARGET_COLUMN: <NONE or 0-6>
COOLDOWN: <READY or BLOCKED>
CONFIDENCE: <0-100>
```

This keeps the model-facing interface simple while preserving a strict deterministic boundary.

```text
probabilistic model output
        ↓
small human-readable protocol
        ↓
deterministic parser
        ↓
strict typed state
```

The parser may tolerate harmless formatting differences, but it does not tolerate ambiguous meaning.

Responses are rejected when they contain:

* a missing required field;
* conflicting values for one field;
* a tank column outside `0..6`;
* a target column outside `0..6`;
* an invalid target-absence marker;
* an unsupported cooldown value;
* a confidence value outside the declared range;
* content that cannot be mapped unambiguously into the policy state.

The model is not trusted merely because it produced syntactically valid text.

---

## Arcade state space

The bounded arcade fixture contains seven horizontal columns.

Each policy state consists of:

```text
tank column
target column or target absence
cooldown state
```

The complete state surface is:

```text
7 tank columns
×
8 target states
×
2 cooldown states
=
112 policy rows
```

The eight target states are:

```text
target absent
target in column 0
target in column 1
target in column 2
target in column 3
target in column 4
target in column 5
target in column 6
```

The two cooldown states are:

```text
0 = ready
1 = blocked
```

The candidate actions are:

```text
LEFT
RIGHT
STAY
FIRE
```

The policy behavior is bounded and explicit:

* no target → `STAY`;
* target aligned and weapon ready → `FIRE`;
* target left of the tank → `LEFT`;
* target right of the tank → `RIGHT`;
* target aligned and weapon blocked → `STAY`.

These rules are compiled into the VPM policy before the visual model is invoked.

---

## Files

Executable example:

```text
examples/local_model_zero_arcade_test.py
```

Living example documentation:

```text
docs/examples/local_model_zero_arcade_test.md
```

Recorded labelled smoke result:

```text
docs/results/local-model-zero-arcade-smoke-v1/README.md
```

Local generated output:

```text
local-results/<model>-zero-arcade-<timestamp>/
├── images/
├── cases.jsonl
├── summary.json
└── run-manifest.json
```

The generated `local-results/` directory is environment-specific evidence and should not normally be committed.

A curated result may instead be preserved under `docs/results/` with:

* the exact command;
* model identity;
* model digest;
* runtime version;
* ZeroModel commit;
* summarized metrics;
* explicit claim boundaries.

---

## Requirements

The example requires:

* a local ZeroModel checkout;
* the repository development dependencies;
* Pillow;
* Ollama;
* a locally installed vision-capable model.

Install the repository dependencies:

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install pillow
```

Check the local Ollama runtime:

```powershell
ollama --version
ollama list
```

Inspect the intended model:

```powershell
ollama show qwen3.5:latest
```

The selected model must support image input.

The example is local-only and is not intended for GitHub Actions.

---

## Verify the harness first

Before invoking a real local model, run the fake provider:

```powershell
python .\examples\local_model_zero_arcade_test.py `
  --backend fake `
  --mode smoke `
  --render labelled
```

The fake provider is a wiring check.

It verifies:

* frame generation;
* state construction;
* policy compilation;
* row addressing;
* action lookup;
* result serialization;
* summary generation.

Expected result:

```text
8 attempted
8 accepted
8 exact states
8 correct actions
0 rejections
```

The fake provider does not perform perception and must not be cited as visual-model evidence.

---

## Run the local-model smoke test

Use the exact model tag reported by `ollama list`.

Example:

```powershell
python .\examples\local_model_zero_arcade_test.py `
  --backend ollama `
  --model qwen3.5:latest `
  --mode smoke `
  --render labelled `
  --confidence-threshold 0.0
```

The smoke suite contains eight representative states covering:

* no visible target;
* both cooldown states;
* an aligned ready target;
* an aligned blocked target;
* targets to the left;
* targets to the right;
* extreme columns;
* intermediate columns.

The confidence threshold is intentionally `0.0` for initial evaluation.

The model’s confidence value is self-reported and should not be treated as calibrated until a separate calibration experiment has been performed.

---

## Labelled and unlabelled render modes

### Labelled mode

```powershell
--render labelled
```

The frame includes:

* lane numbers `0..6`;
* a textual ready or blocked status;
* visible lane boundaries;
* strongly contrasted sprites.

This is the easiest supported perception condition.

It primarily tests whether:

* the model receives the image correctly;
* the text protocol works;
* the parser works;
* the VLM-to-policy boundary works;
* the complete state can address the policy.

A successful labelled run is an integration result, not a robust-vision result.

### Unlabelled mode

```powershell
--render unlabelled
```

The frame retains the seven-lane geometry but removes the printed lane numbers and explanatory status text.

Run:

```powershell
python .\examples\local_model_zero_arcade_test.py `
  --backend ollama `
  --model qwen3.5:latest `
  --mode smoke `
  --render unlabelled `
  --confidence-threshold 0.0
```

This is a more demanding perception test.

It asks whether the model can infer spatial lane positions rather than simply reading the printed labels.

The labelled and unlabelled results must be reported separately.

---

## Complete canonical-state evaluation

After the smoke test behaves correctly, run the complete 112-state surface:

```powershell
python .\examples\local_model_zero_arcade_test.py `
  --backend ollama `
  --model qwen3.5:latest `
  --mode all `
  --render labelled `
  --confidence-threshold 0.0
```

A complete run measures the model over every declared combination of:

* tank column;
* target absence or target column;
* cooldown state.

This is substantially stronger evidence than the eight-state smoke suite.

The complete evaluation should report:

* attempted observations;
* accepted observations;
* rejected observations;
* exact-state accuracy;
* action-equivalent accuracy;
* tank-column accuracy;
* target-presence accuracy;
* target-column accuracy;
* cooldown accuracy;
* response parse failures;
* confidence-threshold rejections;
* latency distribution.

A wrong state can still produce the correct action.

For that reason, exact-state accuracy and action accuracy must remain separate measurements.

---

## Output files

Each execution creates a timestamped local result directory.

### `images/`

Contains the exact rendered frames supplied to the provider.

The image filenames encode the ground-truth state for local evaluation.

The model is not given the filename.

### `cases.jsonl`

Contains one detailed record per observation.

Each record may include:

* image path;
* image digest;
* ground-truth state;
* raw provider response;
* parsed observation;
* confidence;
* predicted row identifier;
* expected row identifier;
* predicted action;
* expected action;
* exact-state result;
* action-equivalent result;
* provider metadata;
* model metadata;
* latency;
* rejection reason.

This is the most important diagnostic output.

When a summary result is surprising, inspect `cases.jsonl` before drawing conclusions.

### `summary.json`

Contains aggregate measurements for the run.

Typical fields include:

```json
{
  "attempted": 8,
  "accepted": 8,
  "rejected": 0,
  "exact_state_correct": 8,
  "action_correct": 8,
  "factor_accuracy_over_accepted": {
    "tank_column": 1.0,
    "target_present": 1.0,
    "target_column": 1.0,
    "cooldown": 1.0
  }
}
```

### `run-manifest.json`

Records the execution arguments and policy artifact identity.

A curated result record should additionally preserve:

* `git rev-parse HEAD`;
* `ollama --version`;
* model tag;
* model digest;
* model parameter count;
* model quantization;
* operating system;
* Python version;
* protocol version;
* prompt identity when available.

---

## Interpreting results

### Exact-state accuracy

Exact-state accuracy requires every state factor to match:

```text
tank column
target presence
target column
cooldown
```

A prediction is exact only when the generated policy row is identical to the ground-truth row.

### Action-equivalent accuracy

Action accuracy measures whether the predicted state addresses a row whose winning action matches the action from the ground-truth state.

For example, several different states may all select `LEFT`.

Therefore:

```text
correct action
```

does not necessarily imply:

```text
correct perception
```

Both metrics are required.

### Factor accuracy

Factor-level measurements identify where perception fails.

Examples:

* high target-presence accuracy but poor target-column accuracy;
* correct spatial positions but unreliable cooldown classification;
* accurate labelled frames but poor unlabelled lane counting.

This is more informative than a single aggregate score.

### Rejections

A rejection is not automatically a system failure.

The adapter should reject observations it cannot map safely into the bounded state vocabulary.

A useful provider may prefer:

```text
correctly reject uncertain observations
```

over:

```text
confidently address the wrong policy row
```

However, rejection behavior must be measured rather than assumed.

### Confidence

The confidence field is generated by the local model.

It is not a probability produced by a calibrated classifier.

Do not interpret:

```text
CONFIDENCE: 95
```

as a verified 95% likelihood of correctness.

Confidence becomes operationally useful only if evaluation shows that it separates correct from incorrect predictions on held-out canonical and perturbed observations.

---

## Validated labelled smoke result

The first recorded local-model smoke experiment used:

| Property        | Recorded value   |
| --------------- | ---------------- |
| Model           | `qwen3.5:latest` |
| Model size      | 9.7B parameters  |
| Quantization    | Q4_K_M           |
| Runtime         | Ollama           |
| Render mode     | Labelled         |
| Cases           | 8                |
| Accepted        | 8                |
| Exact states    | 8                |
| Correct actions | 8                |
| Rejections      | 0                |

The recorded run achieved:

```text
exact-state accuracy: 100%
action accuracy:      100%
tank-column accuracy: 100%
target presence:      100%
target-column accuracy: 100%
cooldown accuracy:    100%
```

The evidence and exact environment identity are preserved at:

```text
docs/results/local-model-zero-arcade-smoke-v1/README.md
```

The strongest supported statement from that result is:

> A local Qwen 3.5 vision model correctly observed all eight labelled canonical arcade smoke-test frames, communicated the complete bounded state through a parsed text protocol, and enabled an independently compiled ZeroModel VPM policy to select the correct action in every case.

That claim is limited to the recorded fixture and environment.

---

## What the example establishes

Within a successful bounded run, the example can establish that:

1. a local visual model received the rendered frame;
2. the model produced image-dependent observations;
3. the observations crossed a small declared protocol;
4. malformed responses could be rejected deterministically;
5. accepted observations could be converted into stable policy rows;
6. an independently identified VPM policy could select the action;
7. the observation provider and policy artifact remained separate;
8. the decision path could retain provider, state, policy, and action evidence.

The architectural result is:

```text
learned perception
        ↓
validated bounded observation
        ↓
identified deterministic policy
```

This is the intended integration seam.

---

## What the example does not establish

This example does not establish:

* general image understanding;
* open-world perception;
* general visual intelligence;
* reliable object detection outside this fixture;
* robustness to arbitrary visual variation;
* real-time gameplay;
* production readiness;
* safety-critical suitability;
* calibrated confidence;
* successful rejection of all unfamiliar frames;
* performance across all 112 states unless the full run is executed;
* unlabelled spatial reasoning unless the unlabelled run is executed;
* successful game trajectories;
* superiority over a conventional detector;
* superiority over direct instrumentation;
* superiority over asking the model to choose an action directly;
* performance across other local models;
* model-independent accuracy.

The canonical labelled frames are:

* synthetic;
* clean;
* strongly contrasted;
* spatially regular;
* generated from the same bounded fixture;
* labelled with the lane indices;
* accompanied by explicit ready or blocked text.

Results must remain attached to the exact tested conditions.

---

## Why this belongs in `examples/`

The local model provider requires:

* an external local runtime;
* a heavyweight model;
* environment-specific hardware;
* nontrivial inference latency;
* model installation outside the ZeroModel packages.

It therefore does not belong in ZeroModel Core.

It also should not introduce an additional production package solely to host one experimental provider.

The example demonstrates how external learned perception can connect to ZeroModel without changing the core artifact contract.

```text
external provider
        ↓
bounded adapter
        ↓
existing ZeroModel policy consumer
```

This keeps the provider replaceable and the policy independently identified.

---

## Why this is not run in CI

The Ollama path is intentionally excluded from routine CI because it requires:

* a locally running Ollama service;
* a downloaded vision-language model;
* several gigabytes of model storage;
* environment-specific CPU or GPU resources;
* seconds of inference time per frame;
* model behavior that may vary across runtime and model versions.

The fake provider can verify the local harness wiring, but it is not visual evidence.

Curated external-model results belong under:

```text
docs/results/
```

with complete environment and claim documentation.

---

## Failure modes

### Model ignores the image

Symptoms:

* identical response for every frame;
* model describes the filename rather than the image;
* selected model does not advertise vision support;
* predictions are unrelated to visible state.

Action:

1. inspect `ollama show <model>`;
2. verify that the model supports vision;
3. run two obviously different images through a standalone image test;
4. ensure the image bytes are included in the Ollama request;
5. do not interpret accidental action matches as perception success.

### Empty response content

Symptoms:

* Ollama completes the request;
* visible response content is empty;
* thinking tokens or schema constraints consume the response path.

Action:

* disable thinking for the request;
* avoid server-side schema-constrained JSON;
* use the small text protocol;
* preserve the raw response metadata for diagnosis.

### Parse rejection

Symptoms:

* model adds prose;
* field names are missing;
* multiple contradictory values appear;
* a field is outside the bounded vocabulary.

Action:

* inspect the raw response;
* improve the prompt only when the problem is systematic;
* do not silently guess missing values;
* keep the parser fail-closed.

### Correct action but incorrect state

Symptoms:

* `action_match` is true;
* `exact_state_match` is false.

Interpretation:

The model supplied the wrong state, but that state happened to select the same action.

This is not exact perception success.

### High confidence and incorrect state

Interpretation:

The self-reported confidence is not calibrated.

Do not raise the confidence threshold until confidence behavior has been evaluated on both correct and incorrect cases.

### First-call latency spike

The first request may include:

* model loading;
* vision encoder initialization;
* memory allocation;
* cache initialization.

Report cold and warmed latency separately when latency becomes part of the claim.

---

## Recommended validation sequence

The example should be advanced through increasingly adversarial stages.

### Stage 1 — labelled smoke

```text
8 representative canonical states
printed lane labels
explicit cooldown text
```

Purpose:

* validate transport;
* validate parsing;
* validate policy addressing;
* validate end-to-end separation.

### Stage 2 — unlabelled smoke

```text
8 representative canonical states
no lane numbers
no explanatory cooldown text
```

Purpose:

* test spatial lane inference;
* reduce dependence on embedded text.

### Stage 3 — complete canonical surface

```text
all 112 declared states
```

Purpose:

* measure the full bounded state space;
* identify factor-specific confusion;
* distinguish exact-state and action-equivalent accuracy.

### Stage 4 — repeatability

Run each state multiple times with pinned settings.

Purpose:

* detect nondeterministic response variation;
* measure policy-address stability;
* compare first-run and repeated-run behavior.

### Stage 5 — controlled perturbations

Introduce one controlled variation at a time:

* brightness;
* contrast;
* sprite colour;
* scale;
* translation;
* blur;
* compression;
* background decoration;
* partial occlusion.

Purpose:

* identify the provider’s operating envelope;
* measure degradation;
* evaluate rejection behavior.

### Stage 6 — trajectory control

Use the local provider to observe successive game states while the compiled policy selects each action.

Purpose:

* measure error accumulation;
* measure trajectory completion;
* inspect how one wrong observation changes later states.

### Stage 7 — direct-action comparison

Compare:

```text
local VLM → predicted state → ZeroModel policy
```

against:

```text
local VLM → direct action
```

Purpose:

* determine whether separating perception from compiled policy improves:

  * state traceability;
  * action consistency;
  * reproducibility;
  * rejection behavior;
  * policy replaceability;
  * auditability.

### Stage 8 — provider comparison

Run the same fixture through:

* direct instrumentation;
* exact deterministic visual reader;
* local VLM provider;
* conventional detector where available;
* additional local vision models.

Purpose:

* determine which provider is appropriate under which deployment assumptions.

---

## Reporting new evidence

Each materially different experiment should receive a separate result record.

Examples:

```text
docs/results/local-model-zero-arcade-unlabelled-smoke-v1/
docs/results/local-model-zero-arcade-canonical-112-v1/
docs/results/local-model-zero-arcade-perturbations-v1/
docs/results/local-model-zero-arcade-trajectory-v1/
docs/results/local-model-zero-arcade-direct-action-comparison-v1/
```

Do not overwrite the original labelled smoke result with stronger later experiments.

Each result record should contain:

* research question;
* exact execution date;
* exact ZeroModel commit;
* exact model tag and digest;
* Ollama version;
* operating system;
* quantization;
* command;
* test surface;
* aggregate metrics;
* important per-case failures;
* latency;
* supported claim;
* unsupported claims;
* links to relevant source and documentation.

This creates an evidence ladder rather than a sequence of undocumented demos.

---

## Design principle

The broader architectural principle demonstrated here is:

> ZeroModel does not need to replace learned perception. It can govern how a probabilistic observation becomes a validated, typed, identity-bearing policy decision.

The provider may change.

The policy artifact may change.

The observation protocol may evolve.

The boundary remains explicit:

```text
observe
    ↓
validate
    ↓
address
    ↓
act
    ↓
record
```

That boundary is the purpose of the example.
