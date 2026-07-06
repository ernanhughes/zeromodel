# Visual Policy Map Artifact Specification v0

Status: **draft**

This document defines the smallest useful ZeroModel artifact.

## 1. Purpose

A Visual Policy Map (VPM) is a deterministic spatial view over a table of scored items.

It exists to make the following information portable and inspectable:

- the numeric values;
- the identity of each row and metric;
- the transformation used to normalize values;
- the row and column ordering used to create the view;
- the provenance of the source data and layout recipe;
- the mapping from rendered cells back to source coordinates.

A VPM does not define an action. A consumer may use a VPM to rank, route, gate, visualize, compare, or audit, but that consumer owns the decision rule.

## 2. Source model

### 2.1 ScoreTable

A `ScoreTable` is a rectangular matrix with stable identifiers.

```text
values:       float[N, M]
row_ids:      string[N]
metric_ids:   string[M]
```

Required invariants:

- `values` is two-dimensional;
- `len(row_ids) == N`;
- `len(metric_ids) == M`;
- row identifiers are unique;
- metric identifiers are unique;
- all values are finite unless a declared missing-value policy is present.

Human labels and application metadata may be attached, but they are not substitutes for stable identifiers.

## 3. Layout model

### 3.1 LayoutRecipe

A layout recipe is explicit data, not hidden logic.

```json
{
  "version": "vpm-layout/0",
  "row_order": {
    "kind": "weighted_score",
    "metrics": [
      {"metric_id": "quality", "weight": 0.7, "direction": "desc"},
      {"metric_id": "uncertainty", "weight": 0.3, "direction": "asc"}
    ],
    "tie_break": "row_id"
  },
  "column_order": {
    "kind": "explicit",
    "metric_ids": ["quality", "uncertainty", "novelty"]
  },
  "normalization": {
    "kind": "per_metric_minmax",
    "clip": true
  }
}
```

The reference implementation may support only a small set of recipe kinds. Unknown kinds must fail validation rather than silently falling back.

### 3.2 Determinism

A recipe is deterministic when:

- every sort has an explicit direction;
- every sort has a stable tie-break;
- normalization behavior is fully specified;
- missing values have a declared policy;
- floating-point canonicalization is specified by the codec.

## 4. Artifact model

A `VPMArtifact` contains canonical values plus the derived view.

```text
artifact_id
spec_version
source
  values
  row_ids
  metric_ids
layout_recipe
view
  normalized_values
  row_order          # view index -> source row index
  column_order       # view index -> source metric index
provenance
created_by
```

The artifact should be immutable after validation.

### 4.1 Artifact identity

`artifact_id` is the SHA-256 digest of canonical serialized semantic content:

```text
spec_version
source values and identifiers
layout recipe
view orders
normalization parameters
provenance inputs that are declared identity-bearing
```

Timestamps, file paths, rendering preferences, and transport metadata are not identity-bearing by default.

### 4.2 Source mapping

For any view cell `(view_row, view_column)`, the artifact must resolve:

```text
source_row_index = row_order[view_row]
source_metric_index = column_order[view_column]
row_id = row_ids[source_row_index]
metric_id = metric_ids[source_metric_index]
raw_value = values[source_row_index, source_metric_index]
normalized_value = normalized_values[view_row, view_column]
```

A renderer that cannot preserve or recover this mapping is a visualization export, not a complete VPM artifact.

## 5. Serialization

The v0 reference codec has two forms.

### 5.1 Canonical bundle

The canonical bundle is the source of truth.

```text
manifest.json
values.npy
normalized.npy
```

A future implementation may use another lossless binary container, but canonical ordering and numeric representation must remain specified.

### 5.2 PNG rendering

PNG is a rendering and transport option.

A PNG may embed the manifest, hashes, and source mapping. It must not be treated as the only semantic representation unless it can round-trip the complete artifact losslessly.

Lossy image transforms invalidate the artifact unless a separate robust watermark specification explicitly says otherwise.

## 6. Inspection API

The minimal conceptual API is:

```python
artifact = build_vpm(score_table, recipe)
artifact.validate()
artifact.cell(view_row=0, view_column=0)
artifact.region(rows=slice(0, 4), columns=slice(0, 4))
artifact.to_bundle(path)
artifact.render_png(path)
loaded = VPMArtifact.from_bundle(path)
```

The artifact API does not include `decide`, `act`, `route`, or arbitrary SQL execution.

Consumers may implement those operations separately:

```python
result = TopLeftGate(threshold=0.8).evaluate(artifact)
```

## 7. Provenance

Minimum provenance fields:

```json
{
  "source_id": "scores/run-0042",
  "source_digest": "sha256:...",
  "producer": {
    "name": "example-scorer",
    "version": "1.3.0"
  },
  "recipe_digest": "sha256:...",
  "parents": []
}
```

Provenance records how the artifact was produced. It does not prove that the source model was correct or that the metrics are meaningful.

## 8. Composition

Composition is permitted only when semantics are declared.

For example, fuzzy intersection may be defined as element-wise minimum over two artifacts only when:

- row identifiers align;
- metric identifiers align;
- normalization domains are compatible;
- the resulting recipe records the operation and both parent artifact IDs.

The words `AND`, `OR`, `NOT`, and `XOR` should not be used without specifying whether the operation is Boolean, fuzzy, probabilistic, set-based, or application-specific.

## 9. Hierarchy

A hierarchical VPM is an index of VPM summaries.

Each parent cell must declare:

- the child artifact or region it summarizes;
- the aggregation function;
- the covered row and metric identifiers;
- whether the aggregation is exact or lossy.

Traversal complexity and latency are benchmark results, not properties guaranteed by this specification.

## 10. Contract tests

The reference implementation must include tests for:

1. deterministic artifact IDs;
2. stable tie-breaking;
3. lossless canonical round-trip;
4. exact cell-to-source mapping;
5. validation failure on duplicate identifiers;
6. validation failure on non-finite values without a policy;
7. recipe failure on unknown metrics;
8. composition failure on incompatible artifacts;
9. rendering that does not mutate semantic content;
10. provenance parent preservation.

## 11. Open questions

The following are intentionally unresolved in v0:

- canonical binary container;
- quantized artifacts;
- streaming append semantics;
- cryptographic signatures;
- robust watermarking;
- hierarchical storage resolvers;
- learned layout recipes.

Each should be added only after the base artifact remains understandable and testable.
