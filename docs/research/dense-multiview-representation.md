# Dense multi-view representation

ZeroModel's core claim is not that a VPM is a heatmap. The claim is that an AI system can externalize dense state into a deterministic artifact that supports many views without regenerating the evidence.

An image contains trees, grass, people, cars, roads, and sky at the same time. Looking for people does not create a new image; it changes what becomes salient. Looking for trees does the same. The source remains the same.

A ZeroModel source table works the same way:

```text
rows    = items, claims, checkpoints, traces, frames, candidates
columns = metrics, scores, risks, evidence signals, policy signals
values  = signal intensity
```

A `ViewProfile` is the policy lens:

```text
same dense source table
+ people view
= people-salient VPM

same dense source table
+ risk view
= risk-salient VPM

same dense source table
+ training-progress view
= checkpoint-progress VPM
```

The evidence is not regenerated. The dense field is reorganized.

## Why this matters

Most LLM interfaces are serial and sparse from the outside. They emit tokens, tool calls, retrieved chunks, summaries, or individual judgments. ZeroModel makes the intermediate state dense and external:

```text
many rows
x many metrics
x deterministic layout
x source mapping
x provenance
```

This enables operations that should not require another model call:

- inspect the top-left decision region
- route from a prepared artifact
- compare two policy views
- preserve source/cell mappings
- replay the same view deterministically
- render or bundle the artifact

## View as policy

A view profile is a small policy over metric salience.

```python
ViewProfile(
    name="hallucination-risk",
    metric_weights={
        "semantic_drift": 1.0,
        "evidence_gap": 0.9,
        "citation_gap": 0.8,
        "verifiability": -0.6,
    },
)
```

Positive weights mean high values matter. Negative weights mean low values matter.

The resulting VPM makes the policy spatial: items matching that policy move toward the inspection region.

## Relation to spatial calculus

The older spatial-calculus tools point beyond manual view profiles. `SpatialOptimizer` learns metric weights to maximize top-left decision mass, computes metric interaction graphs, and derives canonical layouts. `DecisionManifold` extends VPMs over time with curvature, inflection points, critical regions, and decision-flow paths.

Those tools should be treated as the next research layer. `ViewProfile` is the core abstraction they need to target.

## Research question

The first research question is:

> Can dense multi-view artifacts improve human inspection and control of AI system state compared with sparse serial summaries or raw score tables?

This is broader than hallucination detection. Hallucination risk, training progress, learning traces, critic outputs, anomaly detection, and search triage are all views over dense scored state.

## Claim boundary

Valid wording:

> ZeroModel can build multiple deterministic policy views from the same dense source table while preserving source mapping and provenance.

Invalid wording until separately proven:

> ZeroModel automatically learns the best view for every task.

That requires optimized view learning, benchmark tasks, and human inspection studies.
