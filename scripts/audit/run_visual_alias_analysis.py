"""Generate compact codeword-alias evidence for the arcade visual reader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_visual_sign_reader import (  # noqa: E402
    arcade_visual_feature_spec,
    compile_visual_index_artifact,
    enumerate_visual_frames,
)
from zeromodel.core.policy_lookup import VPMPolicyLookup  # noqa: E402
from zeromodel.video.arcade_policy import (  # noqa: E402
    ACTIONS,
    ShooterConfig,
    compile_policy_artifact,
)
from zeromodel.vision import VisualAcceptanceProfile, VisualSignReader  # noqa: E402


def _variants(frame: np.ndarray) -> dict[str, np.ndarray]:
    variants = {
        "canonical_copy": frame.copy(),
        "add_one": np.clip(frame.astype(np.int16) + 1, 0, 255).astype(np.uint8),
        "cooldown_band_removed": frame.copy(),
        "target_column_removed": frame.copy(),
    }
    variants["cooldown_band_removed"][-2:, :] = 0
    variants["target_column_removed"][:, frame.shape[1] // 2] = 0
    return variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = ShooterConfig()
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    index = compile_visual_index_artifact(config, policy_artifact=policy)
    reader = VisualSignReader(index.artifact, policy, action_metric_ids=ACTIONS)
    frames = enumerate_visual_frames(config)
    spec = arcade_visual_feature_spec(config)

    rows = []
    for row_id in list(policy.source.row_ids)[:16]:
        intended_action = lookup.choose(row_id)
        for variant_name, frame in _variants(frames[row_id]).items():
            decision = reader.read(
                frame,
                acceptance_profile=VisualAcceptanceProfile.CALIBRATED_NEAREST,
            )
            addressed_action = None
            if decision.accepted and decision.matched_row_id is not None:
                addressed_action = lookup.choose(decision.matched_row_id)
            rows.append(
                {
                    "source_intended_row": row_id,
                    "variant": variant_name,
                    "addressed_row": decision.matched_row_id,
                    "source_intended_action": intended_action,
                    "addressed_action": addressed_action,
                    "same_action": addressed_action == intended_action,
                    "raw_input_digest": decision.raw_input_digest,
                    "canonical_input_digest": decision.canonical_input_digest,
                    "nearest_canonical_input_digest": decision.nearest_input_digest,
                    "feature_digest": decision.feature_digest,
                    "exact_feature_match": decision.exact_feature_match,
                    "canonical_input_match": decision.canonical_input_match,
                    "acceptance_profile": decision.acceptance_profile,
                    "accepted": decision.accepted,
                    "reason": decision.reason,
                }
            )
    result = {
        "feature_spec_digest": spec.digest,
        "evaluated_rows": 16,
        "variant_count": 4,
        "records": rows,
        "summary": {
            "accepted_noncanonical_aliases": sum(
                int(item["accepted"] and not item["canonical_input_match"])
                for item in rows
            ),
            "action_changing_aliases": sum(
                int(
                    item["accepted"]
                    and not item["canonical_input_match"]
                    and item["addressed_action"] != item["source_intended_action"]
                )
                for item in rows
            ),
        },
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
