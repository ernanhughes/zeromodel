# Research readiness pass

This document describes the pre-research hardening path for ZeroModel.

The goal is not to add another abstract API. The goal is to make the current stack reproducible enough that research claims can be tested against committed fixtures and end-to-end examples.

## What is now demonstrable

ZeroModel can now run this chain:

```text
tracker export fixture
  -> TrainingCheckpoint records
  -> training progress VPM
  -> .vpm bundle
  -> PNG/SVG render
  -> JSON summary
```

The fixtures in `tests/fixtures/training/` cover the same synthetic training run through TensorBoard-style scalar CSV, W&B-style JSONL, Trackio-style nested JSON, and generic JSONL.

The synthetic run deliberately improves until step 3000 and then weakens at step 4000. A correct assessment should select `step_3000` as best and warn that the latest checkpoint is not best.

## End-to-end examples

```bash
python examples/end_to_end_training_progress.py
python examples/end_to_end_learning_trace.py
```

The scripts write outputs under `.zeromodel-demo/`:

```text
.zeromodel-demo/
  training_progress/
    training_progress.vpm
    training_progress.png
    training_progress.svg
    training_progress_summary.json
  learning_trace/
    learning_trace.vpm
    learning_trace.png
    learning_trace.svg
    learning_trace_summary.json
```

These examples are intentionally small. They are not scale benchmarks. They are reproducibility checks showing that artifacts can be generated, rendered, bundled, and summarized from stable inputs.

## Research questions this enables

1. Do training-progress VPMs make checkpoint selection errors easier to detect than scalar dashboards alone?
2. Do learning-trace VPMs help distinguish real transfer from train-only score movement?
3. Do VPM cell/source mappings reduce time-to-debug when a model regresses?
4. Does top-left concentration help reviewers inspect the highest-risk evidence faster?

## Claims still out of scope

Do not use these fixtures to claim:

- planet-scale traversal
- tiny-device timing
- constant-time decision over arbitrary corpora
- PNG survival through lossy image pipelines
- internal model causality

Those require separate benchmark harnesses and hardware profiles.
