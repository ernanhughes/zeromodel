# Controlled PNG Representation Benchmark — Real Ollama Smoke Evidence v1

## Status

**Result status:** `completed_negative_result`  
**Measurement scope:** eight-state synthetic smoke fixture  
**Real provider:** local Ollama `qwen3.5:latest`  
**Canonical 112-state run:** not run  
**Independent repetitions:** not run

This evidence package records a controlled local experiment asking whether changing only the PNG representation improves an unchanged visual provider against an unchanged compiled ZeroModel policy.

The experiment validates the real-provider measurement path under bounded conditions. It does **not** establish general visual understanding, production safety, confidence calibration, cross-model generality, or natural-image performance.

## Execution identity

| Property | Value |
|---|---|
| ZeroModel commit | `3e3753a557bdd21030157f5f96c0a13ac7f56989` |
| Python | `3.11.4` |
| Ollama | `0.32.1` |
| Model | `qwen3.5:latest` |
| Ollama model blob digest | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` |
| Parameters / quantization | `9.7B` / `Q4_K_M` |
| Provider configuration | `sha256:8f878520d89c432ff73073849150e1423b281cebbe4389de2ddd631738b8a4e0` |
| Configuration model digest | `sha256:b8571dca341cc2f9ee4365b30ccaecae43617284b5e44630b8985a12cc1576a7` |
| Prompt digest | `sha256:356a657aecb1b3c7e3b5e13eb830ae24ceeb76acc2156dfd771ea8921375efda` |
| Protocol | `zeromodel-local-model-arcade-text-state/v2` |
| Policy artifact | `sha256:c27d05243c6f960ba30b84fcfe0ddf67c03ee5de053821e29091694caf9d5d4a` |
| Seed / temperature | `0` / `0.0` |
| Confidence threshold | `0.0` |

The configuration model digest above is the ZeroModel identity for the declared model configuration. It is distinct from the Ollama model-blob digest.

## Historical-control interpretation

The labelled control reproduced the prior observable result—`8/8` exact and `8/8` action-correct—with the same Ollama version and model blob digest. The current compiled policy artifact differs from the older recorded policy artifact, so this is a **behavioural reproduction under the same model/runtime**, not a byte-identical reproduction of the entire historical experiment.

## Results

| Variant | Exact | Action-equivalent | Action-changing | Rejected | Action-correct | Classification |
|---|---:|---:|---:|---:|---:|---|
| `labelled-v1` | 8 | 0 | 0 | 0 | 8 | advance relative to unlabelled control |
| `unlabelled-v1` | 3 | 4 | 1 | 0 | 7 | baseline |
| `cooldown-shape-v1` | 4 | 3 | 1 | 0 | 7 | no material change |
| `cooldown-dual-v1` | 3 | 4 | 1 | 0 | 7 | no material change |
| `cooldown-redundant-v1` | 3 | 3 | 2 | 0 | 6 | regression |
| `lane-enhanced-v1` | 2 | 3 | 2 | 1 | 5 | regression |

### Factor accuracy

| Variant | Cooldown | Tank column | Target column | Target present |
|---|---:|---:|---:|---:|
| `labelled-v1` | 8/8 | 8/8 | 8/8 | 8/8 |
| `unlabelled-v1` | 4/8 | 5/8 | 6/8 | 8/8 |
| `cooldown-shape-v1` | 4/8 | 5/8 | 7/8 | 8/8 |
| `cooldown-dual-v1` | 4/8 | 5/8 | 5/8 | 6/8 |
| `cooldown-redundant-v1` | 4/8 | 5/8 | 5/8 | 5/8 |
| `lane-enhanced-v1` | 4/7 | 4/7 | 4/7 | 7/7 |

The lane denominator is seven because one response was rejected before a valid predicted state could be constructed.

## Findings

1. The labelled control was perfect on the bounded smoke fixture.
2. The unlabelled representation preserved action correctness in seven cases despite only three exact states, demonstrating the practical distinction between exact perception and policy-level behaviour.
3. All tested cooldown interventions remained at `4/8` cooldown correctness. The shape intervention's fourth exact case came from better target-column recovery, not the intended cooldown factor.
4. Redundant cooldown markers worsened target-presence recovery and doubled action-changing errors.
5. Lane separator enhancement emphasized eight boundaries around seven lane regions. The provider returned `TANK_COLUMN: 7` for the far-right tank, producing one rejection, and showed several right-shifted coordinate readings.
6. No implicit intervention qualified for combination, repetition, or canonical promotion under the declared gates.

## Bounded conclusion

For local Ollama `qwen3.5:latest` with the recorded model digest, Ollama `0.32.1`, the fixed prompt and parser, the fixed compiled policy, and the eight-state smoke fixture, the tested implicit shape, dual, redundant, and separator-enhanced PNG interventions did not improve policy-level reliability over `unlabelled-v1`. The redundant cooldown and lane-enhanced variants materially regressed.

The experiment supports a narrower positive observation: explicit semantic labels produced perfect bounded recovery, while the tested non-textual redundancy did not. This motivates a later factorial ablation of footer geometry, lane numerals, and cooldown text; it does not yet establish which labelled component caused the improvement.

## Why no canonical run followed

The benchmark's promotion gate requires a smoke candidate to classify as `advance`, or at minimum remain non-regressive while improving its targeted factor. No implicit intervention improved its declared target. Running 112 states would therefore spend provider calls on candidates that failed the predefined smoke screen.

## Evidence layout

- `experiment.json` is the final resumed invocation manifest.
- `comparison.*` contains the generated cross-variant comparison.
- Each variant directory retains its run, recipe, summary, complete case projection and compiled-report identity.
- `selected-images/` contains consequence-focused PNG evidence.
- `curation.json` records omissions and resume-related bookkeeping.
- The original SQLite database and complete PNG set remain in the preserved local source archive.

## Limitations

- Eight synthetic fixture states only.
- One local model blob and one Ollama runtime.
- One prompt, parser and inference configuration.
- No independent repetitions.
- No 112-state canonical execution.
- Temperature zero and a fixed seed do not prove cross-hardware determinism.
- Confidence values were self-reported by the provider and were not calibrated.
- No natural-image, open-world, production-safety or cross-model claim is supported.
