from __future__ import annotations

from pathlib import Path

from zeromodel import TrainingCheckpoint, build_training_progress_vpm, from_bundle, to_bundle, write_png, write_svg


def main() -> None:
    checkpoints = [
        TrainingCheckpoint(
            step=1000,
            metrics={
                "train_loss": 1.00,
                "heldout_score": 0.50,
                "regression_safety": 0.99,
                "stability": 0.96,
                "tokens_per_second": 1000,
            },
        ),
        TrainingCheckpoint(
            step=2000,
            metrics={
                "train_loss": 0.82,
                "heldout_score": 0.57,
                "regression_safety": 0.98,
                "stability": 0.94,
                "tokens_per_second": 1120,
            },
        ),
        TrainingCheckpoint(
            step=3000,
            metrics={
                "train_loss": 0.67,
                "heldout_score": 0.65,
                "regression_safety": 0.97,
                "stability": 0.91,
                "tokens_per_second": 1260,
            },
        ),
        TrainingCheckpoint(
            step=4000,
            metrics={
                "train_loss": 0.58,
                "heldout_score": 0.63,
                "regression_safety": 0.89,
                "stability": 0.84,
                "tokens_per_second": 1240,
            },
        ),
    ]

    assessment = build_training_progress_vpm(
        checkpoints,
        stability_metric="stability",
        efficiency_metric="tokens_per_second",
    )
    artifact = assessment.artifact

    out_dir = Path(".zeromodel-training-demo")
    out_dir.mkdir(exist_ok=True)
    bundle_path = to_bundle(artifact, out_dir / "training_progress.vpm")
    png_path = write_png(artifact, out_dir / "training_progress.png")
    svg_path = write_svg(artifact, out_dir / "training_progress.svg")
    loaded = from_bundle(bundle_path)

    print("learned:", assessment.learned)
    print("best_checkpoint:", assessment.best_checkpoint_id)
    print("best_step:", assessment.best_step)
    print("warnings:", list(assessment.warnings))
    print("artifact_id:", artifact.artifact_id)
    print("bundle_roundtrip:", loaded.artifact_id == artifact.artifact_id)
    print("wrote:", bundle_path, png_path, svg_path)


if __name__ == "__main__":
    main()
