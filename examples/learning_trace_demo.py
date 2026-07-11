from __future__ import annotations

from pathlib import Path

from zeromodel import LearningObservation, build_learning_vpm, from_bundle, to_bundle, write_png, write_svg


def main() -> None:
    observations = [
        LearningObservation(
            unit_id="claim-support",
            before=0.42,
            after=0.72,
            split="train",
            metadata={"feedback": "added missing source citation"},
        ),
        LearningObservation(
            unit_id="related-claim-support",
            before=0.50,
            after=0.63,
            split="heldout",
            metadata={"check": "new claim not used in feedback"},
        ),
        LearningObservation(
            unit_id="unrelated-summary-quality",
            before=0.82,
            after=0.81,
            split="regression",
            metadata={"check": "previously good behavior stayed intact"},
        ),
    ]

    assessment = build_learning_vpm(observations)
    artifact = assessment.artifact

    out_dir = Path(".zeromodel-learning-demo")
    out_dir.mkdir(exist_ok=True)
    bundle_path = to_bundle(artifact, out_dir / "learning_trace.vpm")
    png_path = write_png(artifact, out_dir / "learning_trace.png")
    svg_path = write_svg(artifact, out_dir / "learning_trace.svg")
    loaded = from_bundle(bundle_path)

    print("learned:", assessment.learned)
    print("train_delta:", round(assessment.train_delta, 3))
    print("heldout_delta:", round(assessment.heldout_delta, 3))
    print("regression_degradation:", round(assessment.regression_degradation, 3))
    print("artifact_id:", artifact.artifact_id)
    print("bundle_roundtrip:", loaded.artifact_id == artifact.artifact_id)
    print("wrote:", bundle_path, png_path, svg_path)


if __name__ == "__main__":
    main()
