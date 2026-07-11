from __future__ import annotations

from pathlib import Path

from zeromodel import build_training_progress_vpm, from_bundle, to_bundle, write_png, write_svg
from zeromodel.adapters import checkpoints_from_tensorboard_scalars


def main() -> None:
    out_dir = Path(".zeromodel-tracker-demo")
    out_dir.mkdir(exist_ok=True)
    export_path = out_dir / "tb_scalars.csv"
    export_path.write_text(
        "wall_time,step,tag,value\n"
        "1,1000,train/loss,1.00\n"
        "1,1000,eval/accuracy,0.50\n"
        "1,1000,regression_safety,0.99\n"
        "2,2000,train/loss,0.82\n"
        "2,2000,eval/accuracy,0.57\n"
        "2,2000,regression_safety,0.98\n"
        "3,3000,train/loss,0.70\n"
        "3,3000,eval/accuracy,0.55\n"
        "3,3000,regression_safety,0.92\n",
        encoding="utf-8",
    )

    checkpoints = checkpoints_from_tensorboard_scalars(export_path)
    progress = build_training_progress_vpm(checkpoints)
    artifact = progress.artifact

    bundle_path = to_bundle(artifact, out_dir / "tracker_progress.vpm")
    png_path = write_png(artifact, out_dir / "tracker_progress.png")
    svg_path = write_svg(artifact, out_dir / "tracker_progress.svg")
    loaded = from_bundle(bundle_path)

    print("checkpoints:", [checkpoint.checkpoint_id for checkpoint in checkpoints])
    print("best_checkpoint_id:", progress.best_checkpoint_id)
    print("learned:", progress.learned)
    print("warnings:", progress.warnings)
    print("artifact_id:", artifact.artifact_id)
    print("bundle_roundtrip:", loaded.artifact_id == artifact.artifact_id)
    print("wrote:", bundle_path, png_path, svg_path)


if __name__ == "__main__":
    main()
