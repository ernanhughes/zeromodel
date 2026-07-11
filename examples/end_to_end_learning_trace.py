from __future__ import annotations

import json
from pathlib import Path

from zeromodel import LearningObservation, build_learning_vpm, from_bundle, to_bundle, write_png, write_svg


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / ".zeromodel-demo" / "learning_trace"


def main() -> None:
    assessment = build_learning_vpm(
        [
            LearningObservation(
                unit_id="citation-repair",
                before=0.42,
                after=0.74,
                split="train",
                metadata={"feedback": "missing support citation was added"},
            ),
            LearningObservation(
                unit_id="related-claim-support",
                before=0.50,
                after=0.64,
                split="heldout",
                metadata={"check": "new related claim not directly corrected"},
            ),
            LearningObservation(
                unit_id="previous-summary-quality",
                before=0.84,
                after=0.83,
                split="regression",
                metadata={"check": "previously strong behavior stayed intact"},
            ),
        ]
    )
    artifact = assessment.artifact

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = to_bundle(artifact, OUT_DIR / "learning_trace.vpm")
    png_path = write_png(artifact, OUT_DIR / "learning_trace.png")
    svg_path = write_svg(artifact, OUT_DIR / "learning_trace.svg")
    summary_path = OUT_DIR / "learning_trace_summary.json"
    summary_path.write_text(json.dumps(assessment.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    loaded = from_bundle(bundle_path)
    print("learned:", assessment.learned)
    print("train_delta:", round(assessment.train_delta, 3))
    print("heldout_delta:", round(assessment.heldout_delta, 3))
    print("regression_degradation:", round(assessment.regression_degradation, 3))
    print("artifact_id:", artifact.artifact_id)
    print("bundle_roundtrip:", loaded.artifact_id == artifact.artifact_id)
    print("wrote:", bundle_path)
    print("wrote:", png_path)
    print("wrote:", svg_path)
    print("wrote:", summary_path)


if __name__ == "__main__":
    main()
