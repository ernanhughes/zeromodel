# Local Model Zero Arcade Smoke Test — v1

## Status

**Result:** Passed
**Evidence type:** Local external-model experiment
**Execution date:** 2026-07-23
**Environment:** Local Windows checkout with Ollama
**ZeroModel policy mode:** Independently compiled VPM policy
**Observation provider:** Local vision-language model
**Committed generated images:** No
**CI status:** Not run in CI

This record documents a bounded local experiment in which a vision-language model observed rendered arcade frames, communicated the visible state through a small text protocol, and allowed an independently compiled ZeroModel policy artifact to select the action.

The experiment is not evidence of general visual perception, production safety, or real-time game control.

---

## Research question

> Can a local vision-language model observe a bounded arcade frame, communicate a valid state address to ZeroModel, and allow an independently compiled VPM policy to select the correct action?

The intended separation was:

```text
rendered arcade frame
        ↓
local vision-language model
        ↓
small text observation protocol
        ↓
deterministic parsing and validation
        ↓
stable ZeroModel policy row
        ↓
VPMPolicyLookup
        ↓
LEFT / RIGHT / STAY / FIRE
```

The vision model was responsible only for observing the frame.

It was not asked to select an action and was not given the policy rules.

ZeroModel converted the parsed observation into a stable policy row and selected the action from the compiled VPM policy artifact.

---

## Implementation

Example:

```text
examples/local_model_zero_arcade_test.py
```

Documentation:

```text
docs/examples/local_model_zero_arcade_test.md
```

Result record:

```text
docs/results/local-model-zero-arcade-smoke-v1/README.md
```

Generated local output:

```text
local-results/qwen3_5_latest-zero-arcade-20260723T134223Z/
├── images/
├── cases.jsonl
├── summary.json
└── run-manifest.json
```

The generated `local-results` directory is local evidence and should not be committed as part of the normal repository source tree.

---

## Environment

| Property                        | Value                                               |
| ------------------------------- | --------------------------------------------------- |
| Operating system                | Windows                                             |
| Local model runtime             | Ollama                                              |
| Model                           | `qwen3.5:latest`                                    |
| Model class                     | Vision-language model                               |
| Model parameters                | 9.7B, based on the inspected local model record     |
| Quantization                    | Q4_K_M, based on the inspected local model record   |
| Python environment              | Repository virtual environment                      |
| Render mode                     | Labelled canonical frames                           |
| Test mode                       | Smoke                                               |
| Cases                           | 8                                                   |
| Confidence threshold            | `0.0`                                               |
| External network required       | No                                                  |
| External model service required | Local Ollama only                                   |
| ZeroModel commit                | `40c1348c64930bdd37b7cc6bfb0817c6c736f7f3` |
| Ollama version                  | `0.32.1`                  |
| Model digest                    | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`   |

The final three identity fields should be filled from the exact local execution environment before treating this result as a frozen reproducibility record.

---

## Command

The experiment was executed from the repository root with:

```powershell
python .\examples\local_model_zero_arcade_test.py `
  --backend ollama `
  --model qwen3.5:latest `
  --mode smoke `
  --render labelled `
  --confidence-threshold 0.0
```

---

## Observation protocol

The model returned a small human-readable protocol rather than schema-constrained JSON:

```text
TANK_COLUMN: <0-6>
TARGET_COLUMN: <NONE or 0-6>
COOLDOWN: <READY or BLOCKED>
CONFIDENCE: <0-100>
```

Example:

```text
TANK_COLUMN: 0
TARGET_COLUMN: 6
COOLDOWN: READY
CONFIDENCE: 95
```

The deterministic adapter parsed that response into:

```text
tank=0|target=6|cooldown=0
```

The compiled ZeroModel policy then selected:

```text
RIGHT
```

The model did not return or select `RIGHT` itself.

### Why plain text was used

Earlier schema-constrained JSON tests produced unreliable or empty response content with the tested local model and Ollama combination.

The successful experiment therefore used:

```text
model-generated labelled text
        ↓
deterministic local parser
        ↓
strict typed validation
```

Malformed, incomplete, contradictory, or out-of-range responses are rejected before policy lookup.

---

## Policy artifact

The experiment compiled the complete bounded arcade policy into the following VPM artifact:

```text
b1a64a9222e1635e49f323761c3389a1fba602fc2ebf79b0ecb3966fc3b6f260
```

The policy covers:

```text
7 tank columns
×
8 target states
×
2 cooldown states
=
112 policy rows
```

The candidate actions are:

```text
LEFT
RIGHT
STAY
FIRE
```

The local model supplied the row address.

`VPMPolicyLookup` supplied the action.

---

## Smoke cases

The smoke suite contained eight representative states.

| Case | Ground-truth state                | Expected action | Model state | Result |
| ---: | --------------------------------- | --------------- | ----------- | ------ |
|    1 | `tank=0\|target=none\|cooldown=0` | `STAY`          | Exact       | Passed |
|    2 | `tank=6\|target=none\|cooldown=1` | `STAY`          | Exact       | Passed |
|    3 | `tank=0\|target=0\|cooldown=0`    | `FIRE`          | Exact       | Passed |
|    4 | `tank=3\|target=3\|cooldown=1`    | `STAY`          | Exact       | Passed |
|    5 | `tank=6\|target=0\|cooldown=0`    | `LEFT`          | Exact       | Passed |
|    6 | `tank=0\|target=6\|cooldown=0`    | `RIGHT`         | Exact       | Passed |
|    7 | `tank=3\|target=1\|cooldown=1`    | `LEFT`          | Exact       | Passed |
|    8 | `tank=2\|target=5\|cooldown=1`    | `RIGHT`         | Exact       | Passed |

The cases cover:

* target absence;
* both cooldown states;
* aligned ready states;
* aligned blocked states;
* target to the left;
* target to the right;
* extreme columns;
* intermediate columns.

---

## Recorded result

```json
{
  "accepted": 8,
  "action_accuracy_over_attempted": 1.0,
  "action_correct": 8,
  "attempted": 8,
  "backend": "ollama",
  "case_mode": "smoke",
  "confidence_threshold": 0.0,
  "exact_state_accuracy_over_attempted": 1.0,
  "exact_state_correct": 8,
  "factor_accuracy_over_accepted": {
    "cooldown": 1.0,
    "tank_column": 1.0,
    "target_column": 1.0,
    "target_present": 1.0
  },
  "latency_ms": {
    "max": 6220.271,
    "mean": 2489.0644375,
    "median": 1959.9836,
    "min": 1884.9169,
    "p95": 6220.271
  },
  "model": "qwen3.5:latest",
  "policy_artifact_id": "b1a64a9222e1635e49f323761c3389a1fba602fc2ebf79b0ecb3966fc3b6f260",
  "rejected": 0,
  "rejection_reasons": {
    "accepted": 8
  },
  "render_mode": "labelled",
  "schema_version": "zeromodel-local-qwen3.5-arcade-text-state/v2"
}
```

---

## Result summary

| Measurement              | Result |
| ------------------------ | -----: |
| Attempted observations   |      8 |
| Accepted observations    |      8 |
| Rejected observations    |      0 |
| Exact state recoveries   |    8/8 |
| Correct policy actions   |    8/8 |
| Exact-state accuracy     |   100% |
| Action accuracy          |   100% |
| Tank-column accuracy     |   100% |
| Target-presence accuracy |   100% |
| Target-column accuracy   |   100% |
| Cooldown accuracy        |   100% |

All eight completed observations produced the exact declared policy state.

All eight resulting policy addresses produced the correct ZeroModel action.

---

## Latency

Recorded model-provider latency:

| Measurement |        Time |
| ----------- | ----------: |
| Minimum     | 1,884.92 ms |
| Median      | 1,959.98 ms |
| Mean        | 2,489.06 ms |
| Maximum     | 6,220.27 ms |
| P95         | 6,220.27 ms |

The first observation took approximately 6.2 seconds.

The remaining warmed observations generally took approximately 1.9–2.1 seconds.

This suggests a significant first-call model or vision-path warm-up cost.

These measurements do not establish real-time suitability. They only describe this local execution environment and model configuration.

---

## What the result establishes

Within the committed bounded fixture and tested local environment, the experiment establishes:

1. A local vision-language model can receive the rendered arcade images.
2. The model can distinguish the eight tested visual states.
3. A small labelled text protocol can communicate the observed state reliably.
4. A deterministic parser can convert the model output into a typed ZeroModel state.
5. The state can address an independently compiled VPM policy.
6. `VPMPolicyLookup` can select the correct action from that policy.
7. The perception provider and policy artifact remain separate.
8. Every tested observation, parsed state, policy address, selected action, and artifact identity can be recorded.

The strongest supported statement is:

> A local Qwen 3.5 vision model correctly observed all eight labelled canonical arcade smoke-test frames, communicated the complete bounded state through a parsed text protocol, and enabled an independently compiled ZeroModel VPM policy to select the correct action in every case.

---

## Architectural interpretation

The experiment validates this provider seam:

```text
local learned perception
        ↓
validated observation contract
        ↓
compiled deterministic policy
```

It does not make the local model part of ZeroModel Core.

The model remains an external observation provider.

ZeroModel remains responsible for:

* parsing and validating the observation boundary;
* constructing the stable row address;
* binding the decision to an identified policy artifact;
* selecting the action deterministically;
* preserving the action trace.

This avoids requiring ZeroModel to implement a general visual-state compiler while still allowing it to consume learned perception.

---

## What the result does not establish

This experiment does not establish:

* general image understanding;
* open-world perception;
* robustness to uncontrolled visual variation;
* unlabelled lane recognition;
* performance across all 112 policy states;
* successful control of a complete game trajectory;
* real-time gameplay;
* calibrated model confidence;
* safe rejection of unfamiliar observations;
* robustness to blur, scaling, colour changes, compression, occlusion, or background changes;
* superiority over a conventional detector;
* superiority over asking the VLM to choose the action directly;
* performance on another local model;
* production readiness;
* suitability for safety-critical deployment.

The frames were:

* synthetic;
* canonical;
* clean;
* strongly contrasted;
* explicitly divided into seven lanes;
* labelled with lane numbers;
* generated from the same test harness used to define ground truth.

The 100% result must therefore remain attached to this precise fixture.

---

## Confidence limitation

The model reported confidence values between `95` and `100`.

These values were self-reported by the model and were not calibrated against a held-out distribution.

They must not be interpreted as statistically meaningful probabilities.

The experiment used:

```text
--confidence-threshold 0.0
```

so no otherwise valid prediction was rejected based on self-reported confidence.

Confidence calibration requires a separate experiment containing both correct and incorrect predictions across canonical and perturbed observations.

---

## Reproduction checklist

Before reproducing:

```powershell
git status --short
git rev-parse HEAD
ollama --version
ollama show qwen3.5:latest
```

Install repository dependencies:

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install pillow
```

Verify the local model is available:

```powershell
ollama list
```

Run the wiring-only fake provider:

```powershell
python .\examples\local_model_zero_arcade_test.py `
  --backend fake `
  --mode smoke `
  --render labelled
```

Run the real local vision provider:

```powershell
python .\examples\local_model_zero_arcade_test.py `
  --backend ollama `
  --model qwen3.5:latest `
  --mode smoke `
  --render labelled `
  --confidence-threshold 0.0
```

Review:

```text
summary.json
cases.jsonl
run-manifest.json
images/
```

The `cases.jsonl` record should be treated as the detailed evidence because it contains:

* image digest;
* ground-truth state;
* raw model text;
* parsed observation;
* confidence;
* predicted row ID;
* expected row ID;
* predicted action;
* expected action;
* model metadata;
* provider duration;
* acceptance or rejection reason.

---

## Next experiments

The immediate next experiment is the same eight-state suite without printed lane labels:

```powershell
python .\examples\local_model_zero_arcade_test.py `
  --backend ollama `
  --model qwen3.5:latest `
  --mode smoke `
  --render unlabelled `
  --confidence-threshold 0.0
```

That test asks whether the model can infer the seven spatial lanes rather than reading their printed labels.

After the unlabelled smoke test, possible extensions are:

1. run all 112 canonical states;
2. repeat each state to test response stability;
3. add deterministic brightness, scale, blur, compression, and translation perturbations;
4. compare exact-state accuracy with action-equivalent accuracy;
5. test whether confidence distinguishes correct from incorrect predictions;
6. compare local-VLM state prediction with direct local-VLM action selection;
7. drive a bounded game trajectory using the learned observation provider;
8. compare Qwen with another local vision model under the same protocol.

Each extension requires its own result record and claim boundary.

---

## Conclusion

The v1 labelled smoke experiment passed.

A local Qwen 3.5 vision model recovered all eight tested arcade states exactly. Its outputs were parsed into stable state addresses, and an independently compiled ZeroModel VPM policy selected all eight correct actions.

The result supports a narrow but important architectural conclusion:

> ZeroModel does not need to solve general perception internally to consume visual observations. A local learned provider can perform perception, while ZeroModel retains deterministic state validation, artifact identity, policy lookup, and action traceability.
