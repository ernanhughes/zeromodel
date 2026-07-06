# ZeroModel rebuild

ZeroModel is being rebuilt from first principles.

The original repository contains several valuable experiments, but the central abstraction accumulated too many responsibilities: normalization, feature engineering, ranking, SQL execution, image encoding, hierarchy, edge protocols, provenance, replay, explainability, storage, and decision policy. The public story then treated all of those experiments as one proven system.

This rebuild keeps the strongest idea and makes it precise.

> **A Visual Policy Map is a deterministic spatial artifact derived from a table of scored items.**
>
> It preserves values, ordering, layout intent, and provenance in a form that both software and humans can inspect.

A VPM is a representation. It does not own the policy that produced the scores, decide what action should be taken, or claim to reproduce a model's hidden reasoning.

## Product definition

ZeroModel v2 will provide four things:

1. **A small VPM specification**
   - score table in
   - explicit layout recipe
   - deterministic artifact out
   - stable metadata and hashes

2. **A reference Python implementation**
   - build, validate, serialize, deserialize, compare, and inspect VPM artifacts
   - no global configuration or import-time side effects
   - NumPy as the only mandatory numerical dependency

3. **Evidence-backed experiments**
   - every performance or compression claim has a reproducible benchmark
   - demonstrations are labelled as demonstrations
   - hypotheses are labelled as hypotheses

4. **A public website that explains the artifact by showing it**
   - an interactive score matrix
   - visible reordering under different layout recipes
   - inspectable top-left regions
   - provenance and deterministic replay examples

## Architectural rule

```text
scores + identifiers + layout recipe
                 |
                 v
          VPM artifact
          /     |      \
         v      v       v
    renderer  inspector  consumer policy
      PNG      browser    router/ranker/gate
```

The consumer policy is deliberately outside the artifact.

## What survives from the original work

The rebuild preserves and retests these ideas:

- score matrices can be reorganized into task-specific spatial views;
- deterministic row and metric order can make important regions cheap to inspect;
- the same artifact can support machine inspection and human visualization;
- provenance should travel with the artifact;
- hierarchical summaries may support bounded navigation when the hierarchy is built and benchmarked honestly;
- visual composition can be useful when its semantics are explicitly defined.

## What is no longer a foundational claim

The following are not assumed to be true merely because a PNG can encode numbers:

- that a VPM is a model's chain of thought;
- that reading one pixel constitutes general AI inference;
- that hierarchy makes retrieval latency independent of data size;
- that arbitrary model state survives crop, resize, JPEG conversion, or other lossy transforms;
- that pixel-wise fuzzy operators automatically provide symbolic reasoning;
- that a visual artifact is inherently explanatory without a declared mapping from cells to source items and metrics.

Any of these may be investigated later, but each requires its own experiment and falsification criteria.

## Target repository shape

```text
zeromodel/
  src/zeromodel/
    artifact.py       # immutable artifact and validation
    layout.py         # explicit layout recipes
    codec.py          # lossless serialization
    render.py         # PNG/SVG rendering
    provenance.py     # lineage and hashes
    inspect.py        # regions, cells, comparisons
  tests/
    contract/
    property/
    regression/
  benchmarks/
  examples/
  docs/spec/
  site/
  legacy/             # retained temporarily for reference, not imported by v2
```

## Delivery sequence

### Phase 0 — establish truth

- freeze the current implementation as legacy;
- publish the v0 artifact contract;
- replace the website's unsupported claims with testable statements;
- create one interactive demonstration that works without the Python package.

### Phase 1 — minimal artifact

- implement `ScoreTable`, `LayoutRecipe`, and immutable `VPMArtifact`;
- deterministic IDs and canonical metadata;
- lossless round-trip tests;
- a deliberately small public API.

### Phase 2 — rendering and inspection

- PNG and SVG renderers;
- cell-to-source lookup;
- region summaries;
- artifact diffing;
- browser viewer.

### Phase 3 — evidence

- benchmark serialization size, decode time, region inspection, and hierarchy traversal;
- compare against NPZ, Parquet, Arrow, and plain JSON where appropriate;
- publish raw benchmark data and hardware details.

### Phase 4 — optional consumers

- edge gate example;
- retrieval/ranking adapter;
- Writer runtime adapter;
- hierarchical index experiment.

These consumers depend on the artifact. The artifact never depends on them.

## Definition of done

ZeroModel v2 is credible when a new developer can:

1. understand the complete data contract in ten minutes;
2. build an artifact from a score table with fewer than ten lines of code;
3. map every rendered cell back to its source row and metric;
4. reproduce the same artifact bytes from the same canonical inputs;
5. run every published benchmark locally;
6. distinguish measured results from hypotheses on the website.
