# ZeroModel executable demonstrations

The demo catalogue turns production examples into executed notebooks, evidence records, static HTML, and pages for `zeromodel.org`.

## Commands

```bash
python -m pip install -r demos/requirements.txt
python scripts/build_demos.py validate
python scripts/build_demos.py all --profile fast
python -m http.server 4173 -d build/site
```

Open `http://localhost:4173/demos/`.

## Source ownership

- `examples/` remains the executable production, benchmark, integration, and research source.
- `demos/notebooks/` explains and exercises those sources.
- `demos/catalog.json` controls published notebooks.
- `demos/example-inventory.json` classifies example entrypoints and records their publication status.
- `docs/results/demos/` stores executed notebooks, HTML, and execution metadata.
- `build/site/` is the deployable static site assembled from `site/` and the executed catalogue.

## Notebook contract

Every public notebook must include:

- What this demonstrates
- Why it matters
- Source and package mapping
- Application
- Boundaries and limitations
- Reproduction record

The catalogue uses the evidence states `defined`, `measured`, and `hypothesis`, and the execution profiles `fast`, `extended`, `external`, and `research`.

Current fast notebooks run from `01` through `09`; `09-tiny-critic`
is the deterministic offline judgement-readout demonstration.

## Publication

The Pages workflow executes the fast catalogue and deploys `build/site`. The separate Executable demos workflow remains the focused evidence check and uploads the generated site plus `docs/results/demos/` as a workflow artifact.

## Adding a demo

1. Keep reusable logic in `examples/` or a production package.
2. Add or update its inventory entry.
3. Create a notebook from `_template.ipynb`.
4. Register the notebook in `catalog.json`.
5. Run `python scripts/build_demos.py validate`.
6. Run the appropriate execution profile.
