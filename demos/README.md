# ZeroModel executable demonstrations

This directory turns production examples into executable explanations, reproducibility records, HTML pages, and a static website catalogue.

```text
production example -> notebook -> executed evidence -> website page
```

The notebooks import the existing implementations under `examples/` and the package APIs. They do not fork application logic.

## Build

```bash
python -m pip install -r demos/requirements.txt
python scripts/build_demos.py validate
python scripts/build_demos.py all --profile fast
python -m http.server 4173 -d build/site
```

Open `http://localhost:4173/demos/`.

## Contract

Every notebook must include:

- what it demonstrates;
- why it matters;
- source and package mapping;
- a concrete application;
- boundaries and limitations;
- a reproduction record.

Every notebook declares an evidence state (`defined`, `measured`, or `hypothesis`) and an execution profile (`fast`, `extended`, `external`, or `research`).

Generated execution records and HTML are written beneath `docs/results/demos/`. The assembled static website is written beneath `build/site/`.

## Adding a demo

1. Keep reusable implementation in `examples/` or a package.
2. Add an entry to `demos/catalog.json`.
3. Copy `demos/notebooks/_template.ipynb`.
4. Match the notebook `zeromodel_demo` metadata to the catalogue.
5. Validate and execute the appropriate profile.
