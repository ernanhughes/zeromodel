# Critic evidence VPM

ZeroModel can turn critic outputs into deterministic risk-first artifacts.

This example is shaped by the Writer critic domain: upstream code extracts features, a critic returns `score`, `label`, `verdict`, and `explanation`, and downstream review needs to know which item deserves inspection first.

ZeroModel does not train or run the critic. It artifactizes the critic/evidence result.

## Minimum observation

A critic observation can include:

| Field | Meaning |
|---|---|
| `critic_score` | Higher means the critic thinks the item is better or safer. |
| `policy_fit` | Whether the claim/action fits the declared policy boundary. |
| `evidence_support` | Whether the available evidence supports the claim. |
| `citation_match` | Whether the citation points to the right supporting evidence. |
| `semantic_drift` | How far the claim has drifted from the evidence. |
| `hallucination_energy` | Optional explicit hallucination-energy score. |
| `verifiability` | Optional explicit score for whether the claim can be checked. |

When `hallucination_energy` or `verifiability` is omitted, ZeroModel derives conservative scores from evidence support, citation match, policy fit, and semantic drift.

## Example

```python
from zeromodel import CriticObservation, build_critic_vpm

assessment = build_critic_vpm([
    CriticObservation(
        item_id="claim_supported",
        critic_score=0.91,
        policy_fit=0.95,
        evidence_support=0.92,
        citation_match=0.94,
        semantic_drift=0.04,
    ),
    CriticObservation(
        item_id="claim_hallucinated",
        critic_score=0.25,
        policy_fit=0.38,
        evidence_support=0.18,
        citation_match=0.20,
        semantic_drift=0.82,
        hallucination_energy=0.86,
        verifiability=0.25,
    ),
])

print(assessment.highest_risk_item_id)
artifact = assessment.artifact
```

The resulting artifact is a normal VPM. You can inspect cells, render PNG/SVG, save a `.vpm` bundle, compare artifacts, or gate the top-left risk region.

## Writer critic result shape

Writer-style line criticism can be converted directly:

```python
from zeromodel import build_critic_vpm, observations_from_critic_lines

observations = observations_from_critic_lines({
    "items": [
        {
            "index": 0,
            "text": "Grounded sentence.",
            "score": 0.82,
            "verdict": "good",
            "features": {"support_score": 0.90},
        },
        {
            "index": 1,
            "text": "Ungrounded sentence.",
            "score": 0.28,
            "verdict": "bad",
            "features": {
                "support_score": 0.20,
                "citation_score": 0.30,
                "semantic_drift": 0.76,
            },
        },
    ]
})

assessment = build_critic_vpm(observations)
```

## Metrics

`build_critic_vpm()` creates these metrics:

| Metric | Meaning |
|---|---|
| `risk_score` | Weighted inspection priority. |
| `hallucination_energy` | Explicit or derived hallucination-energy score. |
| `semantic_drift` | Claim/evidence semantic drift. |
| `critic_risk` | `1 - critic_score`. |
| `policy_gap` | `1 - policy_fit`. |
| `evidence_gap` | `1 - evidence_support`. |
| `citation_gap` | `1 - citation_match`. |
| `verifiability` | Explicit or derived verifiability score. |

## Correct claim

Use this wording:

> ZeroModel can turn critic/evidence/policy scores into deterministic risk-first artifacts for inspection.

Avoid this wording:

> ZeroModel detects hallucinations by itself.

The artifact summarizes scored evidence. It does not replace the critic, verifier, judge, or retrieval system that produced the scores.
