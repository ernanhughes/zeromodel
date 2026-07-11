from __future__ import annotations

import json
from pathlib import Path

from zeromodel import build_training_progress_vpm, from_bundle, to_bundle, write_png, write_svg
from zeromodel.adapters import checkpoints_from_tensorboard_scalars


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "training" / "tensorboard_scalars.csv"
OUT_DIR = ROOT / ".zeromodel-demo" / "training_progress"


def main() -> None:
    checkpoints = checkpoints_from_tensorboard_scalars(FIXTURE)
    assessment = build_training_progress_vpm(
        checkpoints,
        stability_metric="stability",
        efficiency_metric="tokens_per_second",
    )
    artifact = assessment.artifact

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = to_bundle(artifact, OUT_DIR / "training_progress.vpm")
    png_path = write_png(artifact, OUT_DIR / "training_progress.png")
    svg_path = write_svg(artifact, OUT_DIR / "training_progress.svg")
    summary_path = OUT_DIR / "training_progress_summary.json"
    summary_path.write_text(json.dumps(assessment.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    loaded = from_bundle(bundle_path)
    print("best_checkpoint_id:", assessment.best_checkpoint_id)
    print("best_step:", assessment.best_step)
    print("learned:", assessment.learned)
    print("warnings:", ",".join(assessment.warnings) or "none")
    print("artifact_id:", artifact.artifact_id)
    print("bundle_roundtrip:", loaded.artifact_id == artifact.artifact_id)
    print("wrote:", bundle_path)
    print("wrote:", png_path)
    print("wrote:", svg_path)
    print("wrote:", summary_path)


if __name__ == "__main__":
    main()
