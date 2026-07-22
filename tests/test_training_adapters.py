from __future__ import annotations

import json

from zeromodel.analysis.training import build_training_progress_vpm
from zeromodel.analysis.adapters import (
    checkpoints_from_jsonl,
    checkpoints_from_tensorboard_scalars,
    checkpoints_from_trackio_export,
    checkpoints_from_wandb_export,
)


def test_tensorboard_scalar_csv_groups_tags_by_step(tmp_path) -> None:
    path = tmp_path / "tb_scalars.csv"
    path.write_text(
        "wall_time,step,tag,value\n"
        "1,1000,train/loss,1.0\n"
        "1,1000,eval/accuracy,0.50\n"
        "1,1000,regression_safety,0.99\n"
        "2,2000,train/loss,0.80\n"
        "2,2000,eval/accuracy,0.58\n"
        "2,2000,regression_safety,0.98\n",
        encoding="utf-8",
    )

    checkpoints = checkpoints_from_tensorboard_scalars(path)
    progress = build_training_progress_vpm(checkpoints)

    assert [checkpoint.step for checkpoint in checkpoints] == [1000, 2000]
    assert checkpoints[0].metrics["train_loss"] == 1.0
    assert checkpoints[1].metrics["heldout_score"] == 0.58
    assert progress.learned is True


def test_wandb_jsonl_flat_history_rows(tmp_path) -> None:
    path = tmp_path / "wandb-history.jsonl"
    rows = [
        {"_step": 1000, "train/loss": 1.0, "val/accuracy": 0.50, "regression/safety": 0.99},
        {"_step": 2000, "train/loss": 0.82, "val/accuracy": 0.57, "regression/safety": 0.98},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    checkpoints = checkpoints_from_wandb_export(path)

    assert checkpoints[0].checkpoint_id == "step_1000"
    assert checkpoints[1].metrics["train_loss"] == 0.82
    assert checkpoints[1].metrics["heldout_score"] == 0.57
    assert checkpoints[1].metrics["regression_safety"] == 0.98


def test_trackio_nested_json_export(tmp_path) -> None:
    path = tmp_path / "trackio.json"
    path.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {"step": 1000, "metrics": {"train_loss": 1.0, "heldout_score": 0.50, "regression_safety": 0.99}},
                    {"step": 2000, "metrics": {"train_loss": 0.76, "heldout_score": 0.62, "regression_safety": 0.97}},
                ]
            }
        ),
        encoding="utf-8",
    )

    checkpoints = checkpoints_from_trackio_export(path)
    progress = build_training_progress_vpm(checkpoints)

    assert progress.best_checkpoint_id == "step_2000"
    assert progress.artifact.source.metadata["kind"] == "training_progress"


def test_generic_jsonl_adapter_supports_nested_metrics(tmp_path) -> None:
    path = tmp_path / "training.jsonl"
    rows = [
        {"step": 10, "metrics": {"train_loss": 1.2, "heldout_score": 0.4, "regression_safety": 1.0}},
        {"step": 20, "metrics": {"train_loss": 0.9, "heldout_score": 0.45, "regression_safety": 1.0}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    checkpoints = checkpoints_from_jsonl(path)

    assert [checkpoint.step for checkpoint in checkpoints] == [10, 20]
    assert checkpoints[0].metrics["heldout_score"] == 0.4
