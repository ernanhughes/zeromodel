# Training telemetry fixtures

These fixtures are intentionally small, sanitized, and deterministic. They exercise the same underlying training run through several common export shapes:

- `tensorboard_scalars.csv` — scalar rows shaped like `wall_time,step,tag,value`
- `wandb_history.jsonl` — flat W&B-style history rows with `_step`
- `trackio_export.json` — nested Trackio-style checkpoint records
- `generic_training.jsonl` — generic nested `metrics` records

The synthetic run improves from step 1000 to 3000, then overfits/regresses slightly at step 4000. A correct ZeroModel progress assessment should select `step_3000` as the best checkpoint and include `latest_checkpoint_is_not_best` in the warnings.

These fixtures are not benchmark evidence for scale or hardware claims. They are reproducibility fixtures for adapter and end-to-end behavior.
