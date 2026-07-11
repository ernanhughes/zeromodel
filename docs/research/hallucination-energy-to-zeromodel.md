# Hallucination Energy to ZeroModel

This note connects the hallucination/policy-gating work to the current ZeroModel implementation.

The working thesis is:

```text
Hallucination Energy was the scoring idea.
Policy gates were the enforcement idea.
ZeroModel is the artifactization layer.
```

ZeroModel does not decide whether text is true on its own. It takes outputs from a critic, verifier, RAG evaluator, judge, or policy engine and turns those outputs into deterministic Visual Policy Map artifacts.

## Mapping

| Hallucination / critic concept | ZeroModel representation |
|---|---|
| Claim or sentence | VPM row |
| Evidence support | Metric column |
| Citation match | Metric column |
| Semantic drift | Metric column |
| Policy fit | Metric column |
| Hallucination energy | Metric column |
| Verifiability | Metric column |
| Critic score | Metric column, inverted into critic risk |
| Critic explanation | Observation metadata |
| Highest-risk item | VPM row ordered to the top-left |
| Accept/reject/review boundary | Downstream gate or controller |

## Writer critic shape

Writer's critic domain provides a useful source shape:

```text
features -> critic score -> label -> verdict -> explanation
```

It also normalizes multiple surfaces into that shape:

- text criticism
- line criticism
- code criticism
- image criticism
- expert-mode review

ZeroModel's `zeromodel.critic` module is designed around that contract. It accepts Writer-style critic result dictionaries and builds risk-first VPM artifacts without depending on Writer runtime internals.

## First research question

> Can deterministic critic/evidence VPMs make hallucination risk and policy failure easier to inspect than scalar scores or raw claim/evidence tables?

The first version should stay synthetic and reproducible:

```text
claim/evidence/policy fixture
  -> critic observations
  -> risk-first VPM
  -> top-left inspection
  -> bundle/render/summary
```

The next version should use real sanitized Writer critic outputs or RAG evaluator outputs.

## Correct claim

Use:

> ZeroModel can turn critic/evidence/policy scores into deterministic risk-first artifacts for inspection.

Avoid:

> ZeroModel detects hallucinations by itself.

ZeroModel is the artifact layer. It preserves evidence mapping and prioritizes inspection. The scorer, critic, retriever, verifier, or judge remains responsible for producing the scores.
