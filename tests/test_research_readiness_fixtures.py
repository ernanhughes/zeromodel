from __future__ import annotations

from pathlib import Path

from zeromodel.analysis.training import build_training_progress_vpm
from zeromodel.core.bundle import (
    from_bundle,
    to_bundle,
)
from zeromodel.core.render import (
    write_png,
    write_svg,
)
from zeromodel.analysis.adapters import (
    checkpoints_from_jsonl,
    checkpoints_from_tensorboard_scalars,
    checkpoints_from_trackio_export,
    checkpoints_from_wandb_export,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "training"


def _assert_training_fixture_progress(checkpoints) -> None:
    assessment = build_training_progress_vpm(
        checkpoints,
        stability_metric="stability",
        efficiency_metric="tokens_per_second",
    )

    assert len(assessment.checkpoints) == 4
    assert assessment.best_checkpoint_id == "step_3000"
    assert assessment.best_step == 3000
    assert assessment.learned is True
    assert "latest_checkpoint_is_not_best" in assessment.warnings
    assert assessment.artifact.provenance["kind"] == "training_progress"
    assert assessment.artifact.provenance["best_checkpoint_id"] == "step_3000"

    first_cell = assessment.artifact.cell(0, 0)
    assert first_cell.row_id == "step_3000"
    assert first_cell.metric_id == "progress_score"


def test_tensorboard_fixture_reaches_expected_progress() -> None:
    checkpoints = checkpoints_from_tensorboard_scalars(FIXTURE_DIR / "tensorboard_scalars.csv")
    _assert_training_fixture_progress(checkpoints)


def test_wandb_fixture_reaches_expected_progress() -> None:
    checkpoints = checkpoints_from_wandb_export(FIXTURE_DIR / "wandb_history.jsonl")
    _assert_training_fixture_progress(checkpoints)


def test_trackio_fixture_reaches_expected_progress() -> None:
    checkpoints = checkpoints_from_trackio_export(FIXTURE_DIR / "trackio_export.json")
    _assert_training_fixture_progress(checkpoints)


def test_generic_jsonl_fixture_reaches_expected_progress() -> None:
    checkpoints = checkpoints_from_jsonl(FIXTURE_DIR / "generic_training.jsonl")
    _assert_training_fixture_progress(checkpoints)


def test_training_fixture_round_trips_and_renders(tmp_path: Path) -> None:
    checkpoints = checkpoints_from_tensorboard_scalars(FIXTURE_DIR / "tensorboard_scalars.csv")
    assessment = build_training_progress_vpm(
        checkpoints,
        stability_metric="stability",
        efficiency_metric="tokens_per_second",
    )

    bundle_path = to_bundle(assessment.artifact, tmp_path / "training_progress.vpm")
    png_path = write_png(assessment.artifact, tmp_path / "training_progress.png")
    svg_path = write_svg(assessment.artifact, tmp_path / "training_progress.svg")

    loaded = from_bundle(bundle_path)
    assert loaded.artifact_id == assessment.artifact.artifact_id
    assert Path(png_path).exists()
    assert Path(svg_path).exists()
