# ZeroModel

**ZeroModel turns scored data into deterministic, inspectable spatial artifacts.**

The project is being rebuilt from first principles around one narrow abstraction: the **Visual Policy Map (VPM)**.

> A VPM is a deterministic spatial view over a table of scored items.

It carries the numeric values, stable item and metric identifiers, layout recipe, view ordering, source mapping, and provenance needed to inspect how the artifact was formed.

A VPM is **not** a foundation model, a model's hidden chain of thought, or an action policy. Models, rules, sensors, and humans may produce scores. Separate consumers may rank, route, gate, visualize, or audit a VPM. The artifact itself remains a representation.

## Rebuild status

This repository currently contains two layers:

- `zeromodel/` — the original v1 experimental implementation. It is retained for reference while the useful ideas are retested.
- the v2 rebuild documents and website — the new contract being established before a replacement Python package is written.

Do not treat the current v1 package API as the future v2 API.

Read the rebuild plan in [`REBUILD.md`](REBUILD.md) and the draft artifact contract in [`docs/spec/vpm-artifact-v0.md`](docs/spec/vpm-artifact-v0.md).

## The core pipeline

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

The separation matters:

1. **Score producers** own what the metrics mean.
2. **Layout recipes** explicitly define normalization, ordering, direction, and tie-break behavior.
3. **VPM artifacts** preserve the resulting view and its mapping back to source evidence.
4. **Consumers** own any threshold, route, ranking, or action.

## Website

The new website is a zero-dependency static application in `site/`.

It includes an interactive demonstration that:

- uses one fixed source score table;
- applies several declared layout recipes;
- visibly reorganizes rows and metrics;
- measures a configurable top-left region;
- maps every displayed cell back to its source row, source metric, raw value, and normalized value;
- distinguishes specified behavior, measured results, and research hypotheses.

Run it locally:

```bash
python -m http.server 4173 -d site
```

Then open `http://localhost:4173`.

GitHub Pages deployment is defined in [`.github/workflows/pages.yml`](.github/workflows/pages.yml).

## What the rebuild preserves

The original work contains several ideas worth retaining and testing carefully:

- task-specific spatial ordering of score matrices;
- cheap inspection of declared high-priority regions;
- one artifact serving both machine inspection and human visualization;
- provenance travelling with the representation;
- hierarchical summaries for bounded navigation;
- composition when alignment and operator semantics are explicit.

## What must be demonstrated

Claims about compression, latency, edge execution, hierarchy, explainability, or reasoning will be published only when the repository contains:

- the benchmark or experiment code;
- raw output;
- hardware and software details;
- the dataset or generator;
- comparison baselines;
- falsification criteria.

The website uses three evidence labels:

- **Defined** — required by the artifact specification;
- **Measured** — reproduced by a benchmark in this repository;
- **Hypothesis** — proposed research not yet established.

## Target v2 API

The final names may change, but the intended surface is deliberately small:

```python
score_table = ScoreTable(
    values=values,
    row_ids=row_ids,
    metric_ids=metric_ids,
)

recipe = LayoutRecipe.from_dict(recipe_data)
artifact = build_vpm(score_table, recipe)

artifact.validate()
artifact.cell(view_row=0, view_column=0)
artifact.region(rows=slice(0, 4), columns=slice(0, 4))
artifact.to_bundle("artifact.vpm")
artifact.render_png("artifact.png")
```

Decision behavior belongs outside the artifact:

```python
result = TopLeftGate(threshold=0.8).evaluate(artifact)
```

## Delivery order

1. Freeze and document the useful v1 experiments.
2. Finalize the minimal VPM artifact contract.
3. Build the immutable Python reference implementation.
4. Add lossless serialization, rendering, inspection, and artifact diffing.
5. Establish honest format and latency benchmarks.
6. Reintroduce edge, retrieval, hierarchy, and composition as optional consumers or experiments.

## Contributing

The most useful contributions during the rebuild are:

- criticism of the artifact contract;
- small reproducible counterexamples;
- baseline implementations;
- property and round-trip tests;
- benchmark design;
- accessibility and inspection improvements for the website.

## License

MIT.
