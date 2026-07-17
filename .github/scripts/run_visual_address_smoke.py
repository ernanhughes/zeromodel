"""Efficient CI runner for the pinned Phase 1 frozen-address smoke benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import arcade_visual_address_benchmark as demo  # noqa: E402
from zeromodel.visual_encoder import HuggingFaceDinoV2Encoder  # noqa: E402
from zeromodel.visual_experiment import (  # noqa: E402
    build_research_report,
    encode_observations,
    evaluate_visual_provider,
    records_for_split,
    vectors_for_records,
)
from zeromodel.visual_precomputed import PrecomputedVectorAddressProvider  # noqa: E402
from zeromodel.visual_retrieval import (  # noqa: E402
    LinearProbeIndex,
    VectorAddressIndex,
    build_linear_probe,
)


def run(
    *,
    output_dir: Path,
    variants_per_family: int,
    device: str,
    local_files_only: bool,
) -> Dict[str, Any]:
    dataset, baseline_report, artifacts = demo.run_benchmark(
        variants_per_family=variants_per_family,
        encoder_name="none",
        include_traces=False,
    )
    systems = list(baseline_report.systems)

    encoder = HuggingFaceDinoV2Encoder(
        device=device,
        local_files_only=local_files_only,
    )
    all_ids = tuple(record.observation_id for record in dataset.manifest.records)
    vectors = encode_observations(
        encoder,
        all_ids,
        dataset.observations,
        batch_size=32,
    )
    vectors_by_digest = {
        dataset.observations[observation_id].raw_digest: vector
        for observation_id, vector in vectors.items()
    }

    for system_id, strategy, name in (
        ("C", "medoid", "frozen_embedding_medoids"),
        ("D", "all", "raw_embedding_knn"),
    ):
        build, _ = demo._build_vector_system(
            dataset,
            encoder,
            vectors,
            strategy=strategy,
        )
        provider = PrecomputedVectorAddressProvider(
            VectorAddressIndex(build),
            vectors_by_digest,
        )
        result, _ = evaluate_visual_provider(
            provider=provider,
            dataset_manifest=dataset.manifest,
            observations=dataset.observations,
            policy_lookup=dataset.policy_lookup,
            system_id=system_id,
            system_name=name,
            include_traces=False,
        )
        systems.append(result)
        artifacts["address_%s" % system_id] = {
            "manifest": build.manifest.to_dict(),
            "calibration": build.calibration.to_dict(),
            "matrix_blob": build.matrix_blob.to_dict(),
        }

    prototype = vectors_for_records(
        records_for_split(dataset.manifest, "prototype"),
        vectors,
    )
    calibration = vectors_for_records(
        records_for_split(dataset.manifest, "calibration"),
        vectors,
    )
    probe_build = build_linear_probe(
        prototype_vectors=prototype[0],
        prototype_row_ids=prototype[1],
        prototype_action_ids=prototype[2],
        calibration_vectors=calibration[0],
        calibration_row_ids=calibration[1],
        calibration_action_ids=calibration[2],
        policy_artifact_id=dataset.policy.artifact_id,
        source_scope=demo.SOURCE_SCOPE,
        representation_spec_digest=encoder.manifest().manifest_id,
        encoder_manifest_id=encoder.manifest().manifest_id,
    )
    probe_provider = PrecomputedVectorAddressProvider(
        LinearProbeIndex(probe_build),
        vectors_by_digest,
    )
    result_g, _ = evaluate_visual_provider(
        provider=probe_provider,
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        system_id="G",
        system_name="rejection_equipped_linear_probe",
        include_traces=False,
    )
    systems.append(result_g)
    artifacts["encoder_manifest"] = encoder.manifest().to_dict()
    artifacts["linear_probe_G"] = {
        "model_id": probe_build.model_id,
        "weights_blob": probe_build.weights_blob.to_dict(),
        "calibration": probe_build.calibration.to_dict(),
        "row_ids": list(probe_build.row_ids),
        "action_ids": list(probe_build.action_ids),
    }

    report = build_research_report(
        dataset_manifest=dataset.manifest,
        system_results=systems,
        metadata={
            "fixture": "bounded_arcade_shooter",
            "encoder": "dinov2",
            "variants_per_family": variants_per_family,
            "claim_status": "research",
            "execution_plan": "one_frozen_extraction_reused_across_C_D_G",
        },
    )
    payload = {
        "dataset_manifest": dataset.manifest.to_dict(),
        "benchmark_report": report.to_dict(),
        "artifacts": artifacts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "arcade-visual-phase-one.json"
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Use the public to_dict contracts recursively. Calling dict() on the outer
    # immutable notes mapping leaves nested mappingproxy values frozen and caused
    # the original CI summary serialization failure.
    report_dict = report.to_dict()
    summary = {
        "dataset_digest": dataset.manifest.digest,
        "observation_count": len(dataset.manifest.records),
        "report_digest": report.digest,
        "systems": {
            item["system_id"]: {
                "metrics": item["metrics"],
                "notes": item["notes"],
            }
            for item in report_dict["systems"]
        },
        "validation_status": report.validation_status,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants-per-family", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    summary = run(
        output_dir=args.output_dir,
        variants_per_family=args.variants_per_family,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
