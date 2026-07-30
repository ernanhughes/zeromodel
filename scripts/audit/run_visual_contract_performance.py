"""Measure compact VisualSignReader contract-hardening costs."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_visual_sign_reader import (  # noqa: E402
    arcade_visual_feature_spec,
    compile_visual_index_artifact,
    enumerate_visual_frames,
)
from zeromodel.video.arcade_policy import (  # noqa: E402
    ACTIONS,
    ShooterConfig,
    compile_policy_artifact,
)
from zeromodel.vision import (  # noqa: E402
    VisualAcceptanceProfile,
    VisualSignReader,
    extract_visual_features,
    visual_input_digest,
    visual_raw_input_digest,
)


def _measure(repeats: int, fn) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1000.0)
    return {
        "median_us": statistics.median(samples),
        "p95_us": sorted(samples)[int(0.95 * (len(samples) - 1))],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    config = ShooterConfig()
    policy = compile_policy_artifact(config)
    index = compile_visual_index_artifact(config, policy_artifact=policy)
    reader = VisualSignReader(index.artifact, policy, action_metric_ids=ACTIONS)
    spec = arcade_visual_feature_spec(config)
    frame = next(iter(enumerate_visual_frames(config).values()))
    decision = reader.read(frame)

    input_digests = dict(index.artifact.source.metadata["input_digests"])
    artifact_bytes = json.dumps(
        index.artifact.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    result = {
        "environment": {
            "machine": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "repeats": args.repeats,
        "measurements": {
            "feature_extraction": _measure(
                args.repeats, lambda: extract_visual_features(frame, spec)
            ),
            "raw_digest_generation": _measure(
                args.repeats, lambda: visual_raw_input_digest(frame, spec)
            ),
            "canonical_digest_generation": _measure(
                args.repeats, lambda: visual_input_digest(frame, spec)
            ),
            "canonical_only_read": _measure(
                args.repeats,
                lambda: reader.read(
                    frame, acceptance_profile=VisualAcceptanceProfile.CANONICAL_ONLY
                ),
            ),
            "exact_codeword_read": _measure(
                args.repeats,
                lambda: reader.read(
                    frame, acceptance_profile=VisualAcceptanceProfile.EXACT_CODEWORD
                ),
            ),
            "calibrated_nearest_read": _measure(
                args.repeats,
                lambda: reader.read(
                    frame, acceptance_profile=VisualAcceptanceProfile.CALIBRATED_NEAREST
                ),
            ),
            "decision_serialization": _measure(
                args.repeats, lambda: decision.to_dict()
            ),
        },
        "sizes": {
            "visual_index_artifact_json_bytes": len(artifact_bytes),
            "input_digest_metadata_json_bytes": len(
                json.dumps(input_digests, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ),
            "state_count": len(input_digests),
        },
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
