"""Exact canonical temporal video baseline for ZeroModel 1.0.12 research.

This is a positive transport, temporal-governance, and lineage baseline. It does
not measure tolerance to approximate world observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_shooter_policy import (  # noqa: E402
    ACTIONS,
    ShooterConfig,
    TinyArcadeShooter,
    compile_policy_artifact,
    state_row_id,
)
from examples.arcade_visual_sign_reader import (  # noqa: E402
    compile_visual_index_artifact,
    make_visual_reader,
    render_state_frame,
)
from zeromodel import VPMPolicyLookup  # noqa: E402
from zeromodel.policy_transitions import PolicyTransitionSpec  # noqa: E402
from zeromodel.video import InMemoryVideoFrameSource  # noqa: E402
from zeromodel.video_policy import VideoPolicyReader  # noqa: E402
from zeromodel.visual_policy import DeterministicVisualAddressProvider  # noqa: E402


def _parse_row_id(row_id: str) -> Tuple[int, Optional[int], int]:
    values = {}
    for part in str(row_id).split("|"):
        key, value = part.split("=", 1)
        values[key] = value
    target = None if values["target"] == "none" else int(values["target"])
    return int(values["tank"]), target, int(values["cooldown"])


def _next_rows(
    tank_x: int,
    target_x: Optional[int],
    cooldown: int,
    action: str,
    *,
    width: int,
) -> Tuple[str, ...]:
    action = str(action)
    next_tank = tank_x
    if action == "LEFT":
        next_tank = max(0, tank_x - 1)
    elif action == "RIGHT":
        next_tank = min(width - 1, tank_x + 1)

    if action == "FIRE":
        next_cooldown = 1 if cooldown == 0 else cooldown
    else:
        next_cooldown = max(0, cooldown - 1)

    successful_fire = (
        action == "FIRE"
        and cooldown == 0
        and target_x is not None
        and tank_x == target_x
    )
    if successful_fire:
        next_targets: Tuple[Optional[int], ...] = (None,) + tuple(range(width))
    else:
        next_targets = (target_x,)
    return tuple(
        state_row_id(next_tank, next_target, next_cooldown)
        for next_target in next_targets
    )


def arcade_transition_spec(
    config: ShooterConfig = ShooterConfig(),
    *,
    maximum_frame_gap: int = 2,
) -> PolicyTransitionSpec:
    policy = compile_policy_artifact(config)
    transitions: Dict[str, Tuple[str, ...]] = {}
    for row_id in policy.source.row_ids:
        tank_x, target_x, cooldown = _parse_row_id(str(row_id))
        destinations = set()
        for action in ACTIONS:
            destinations.update(
                _next_rows(
                    tank_x,
                    target_x,
                    cooldown,
                    action,
                    width=config.width,
                )
            )
        transitions[str(row_id)] = tuple(sorted(destinations))
    return PolicyTransitionSpec(
        allowed_row_transitions=transitions,
        maximum_frame_gap=maximum_frame_gap,
        maximum_position_delta=1,
        action_conditioned=False,
        metadata={
            "world": "tiny_arcade_shooter",
            "derivation": "declared_environment_dynamics",
        },
    )


def build_canonical_arcade_clip(
    config: ShooterConfig = ShooterConfig(),
    *,
    fps: float = 10.0,
) -> Tuple[InMemoryVideoFrameSource, Tuple[str, ...], Tuple[str, ...]]:
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    game = TinyArcadeShooter(config)
    frames = []
    expected_rows = []
    expected_actions = []
    while not game.done:
        row_id = game.row_id()
        decision = lookup.read(row_id)
        frames.append(
            render_state_frame(
                game.tank_x,
                game.target_x,
                game.cooldown,
                width=config.width,
            )
        )
        expected_rows.append(row_id)
        expected_actions.append(decision.action)
        game.step(decision.action)
    source = InMemoryVideoFrameSource.from_arrays(
        frames,
        clip_id="arcade-canonical-policy-episode",
        nominal_fps=fps,
        source_id="examples.arcade_visual_video_baseline",
        metadata={
            "fixture": "exact_canonical_arcade_video",
            "expected_disposition": "accept",
        },
    )
    return source, tuple(expected_rows), tuple(expected_actions)


def run_exact_video_baseline(
    config: ShooterConfig = ShooterConfig(),
) -> dict[str, Any]:
    policy = compile_policy_artifact(config)
    visual_build = compile_visual_index_artifact(config, policy_artifact=policy)
    visual_reader = make_visual_reader(policy, visual_build)
    provider = DeterministicVisualAddressProvider(
        visual_reader,
        source_scope="fixture:exact-canonical-arcade-video",
    )
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    reader = VideoPolicyReader(
        provider,
        lookup,
        arcade_transition_spec(config),
        evidence_window_size=4,
        maximum_identical_frame_run=3,
    )
    source, expected_rows, expected_actions = build_canonical_arcade_clip(config)
    trace = reader.read(source)
    observed_rows = tuple(decision.accepted_row_id for decision in trace.decisions)
    observed_actions = tuple(decision.accepted_action_id for decision in trace.decisions)
    if trace.accepted_count != trace.manifest.frame_count:
        raise RuntimeError("canonical video baseline rejected a frame")
    if observed_rows != expected_rows:
        raise RuntimeError("canonical video row sequence differs from symbolic policy")
    if observed_actions != expected_actions:
        raise RuntimeError("canonical video action sequence differs from symbolic policy")
    return {
        "system": "V0_exact_canonical_video_reader",
        "policy_artifact_id": policy.artifact_id,
        "visual_index_artifact_id": visual_build.artifact.artifact_id,
        "provider_contract_digest": provider.contract().digest,
        "transition_spec_id": reader.transition_spec.spec_id,
        "clip_manifest_id": trace.manifest.manifest_id,
        "trace_id": trace.trace_id,
        "frame_count": trace.manifest.frame_count,
        "accepted_frames": trace.accepted_count,
        "exact_row_sequence_match": True,
        "action_sequence_match": True,
        "trace": trace.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_exact_video_baseline()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "trace"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
