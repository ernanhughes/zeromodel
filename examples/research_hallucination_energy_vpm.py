from __future__ import annotations

import json
from pathlib import Path

from zeromodel import CriticObservation, build_critic_vpm, from_bundle, to_bundle, write_png, write_svg


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / ".zeromodel-demo" / "critic_evidence"


def main() -> None:
    observations = [
        CriticObservation(
            item_id="claim_supported",
            critic_score=0.91,
            policy_fit=0.95,
            evidence_support=0.92,
            citation_match=0.94,
            semantic_drift=0.04,
            verdict="good",
            metadata={
                "claim": "The cited source states the model improved on the held-out check.",
                "evidence_id": "source_a",
            },
        ),
        CriticObservation(
            item_id="claim_weak_citation",
            critic_score=0.62,
            policy_fit=0.70,
            evidence_support=0.48,
            citation_match=0.35,
            semantic_drift=0.42,
            verdict="risky",
            explanation=[{"feature": "citation_match", "contribution": -0.41}],
            metadata={
                "claim": "The source proves the policy boundary was satisfied.",
                "evidence_id": "source_b",
            },
        ),
        CriticObservation(
            item_id="claim_hallucinated",
            critic_score=0.25,
            policy_fit=0.38,
            evidence_support=0.18,
            citation_match=0.20,
            semantic_drift=0.82,
            hallucination_energy=0.86,
            verifiability=0.25,
            verdict="bad",
            explanation=[{"feature": "semantic_drift", "contribution": -0.77}],
            metadata={
                "claim": "The evidence says the system was validated on real production runs.",
                "evidence_id": "source_c",
                "note": "Synthetic example: claim exceeds the available evidence.",
            },
        ),
    ]

    assessment = build_critic_vpm(observations)
    artifact = assessment.artifact

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = to_bundle(artifact, OUT_DIR / "critic_evidence.vpm")
    png_path = write_png(artifact, OUT_DIR / "critic_evidence.png")
    svg_path = write_svg(artifact, OUT_DIR / "critic_evidence.svg")
    loaded = from_bundle(bundle_path)

    summary = assessment.to_dict()
    summary["bundle_roundtrip"] = loaded.artifact_id == artifact.artifact_id
    summary["outputs"] = {
        "bundle": str(bundle_path.relative_to(ROOT)),
        "png": str(png_path.relative_to(ROOT)),
        "svg": str(svg_path.relative_to(ROOT)),
    }
    summary_path = OUT_DIR / "critic_evidence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("highest_risk_item_id:", assessment.highest_risk_item_id)
    print("highest_risk_score:", round(assessment.highest_risk_score, 3))
    print("warnings:", ",".join(assessment.warnings))
    print("bundle_roundtrip:", summary["bundle_roundtrip"])
    print("wrote:", summary_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
