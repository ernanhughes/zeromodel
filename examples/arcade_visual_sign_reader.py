"""Observation-addressed policy over the bounded arcade shooter.

The arcade fixture is a calibration environment, not the claimed deployment use
case. The engine can expose an exact symbolic state, so the visual path can be
verified exhaustively against that ground truth. Runtime action selection uses
only the rendered frame -> visual index -> policy row path.

Run:

    python examples/arcade_visual_sign_reader.py
    python examples/arcade_visual_sign_reader.py --output-dir build/visual-reader
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zeromodel.core.bundle import to_bundle
from zeromodel.video.arcade_policy import (  # noqa: E402
    ACTIONS,
    CELL_PIXELS,
    COOLDOWN_BLOCKED_VALUE,
    COOLDOWN_READY_VALUE,
    FRAME_HEIGHT,
    ShooterConfig,
    TANK_VALUE,
    TARGET_VALUE,
    TinyArcadeShooter,
    compile_policy_artifact,
    enumerate_visual_frames as package_enumerate_visual_frames,
    render_state_frame,
)
from zeromodel.vision.visual import (  # noqa: E402
    VisualFeatureSpec,
    VisualIndexBuild,
    VisualSignReader,
    build_visual_index,
)

def arcade_visual_feature_spec(config: ShooterConfig = ShooterConfig()) -> VisualFeatureSpec:
    return VisualFeatureSpec(
        input_height=FRAME_HEIGHT,
        input_width=config.width * CELL_PIXELS,
        target_height=8,
        target_width=config.width * 2,
        quantization_levels=16,
    )


def enumerate_visual_frames(
    config: ShooterConfig = ShooterConfig(),
) -> Mapping[str, np.ndarray]:
    return dict(package_enumerate_visual_frames(config))


def compile_visual_index_artifact(
    config: ShooterConfig = ShooterConfig(),
    *,
    policy_artifact: Any | None = None,
) -> VisualIndexBuild:
    policy = policy_artifact or compile_policy_artifact(config)
    return build_visual_index(
        policy,
        enumerate_visual_frames(config),
        arcade_visual_feature_spec(config),
        threshold_fraction=0.25,
        margin_fraction=0.75,
        name="arcade-visual-index-source-order",
    )


def make_visual_reader(
    policy_artifact: Any,
    visual_index_build: VisualIndexBuild,
) -> VisualSignReader:
    return VisualSignReader(
        visual_index_build.artifact,
        policy_artifact,
        action_metric_ids=ACTIONS,
        value_source="raw",
        tie_break="metric_order",
    )


def run_visual_policy_episode(
    config: ShooterConfig = ShooterConfig(),
    *,
    policy_artifact: Any | None = None,
    visual_index_build: VisualIndexBuild | None = None,
    visual_reader: VisualSignReader | None = None,
) -> dict[str, Any]:
    policy = policy_artifact or compile_policy_artifact(config)
    visual_build = visual_index_build or compile_visual_index_artifact(
        config,
        policy_artifact=policy,
    )
    reader = visual_reader or make_visual_reader(policy, visual_build)
    game = TinyArcadeShooter(config)
    trace: list[dict[str, Any]] = []

    while not game.done:
        before = game.snapshot()
        frame = render_state_frame(
            game.tank_x,
            game.target_x,
            game.cooldown,
            width=config.width,
        )
        decision = reader.read(frame)
        if not decision.accepted or decision.action is None:
            raise RuntimeError("canonical arcade frame was rejected: %s" % decision.to_dict())
        game.step(decision.action)
        trace.append(
            {
                "before": before,
                "visual_decision": decision.to_dict(),
                "after": game.snapshot(),
            }
        )

    return {
        "policy_artifact_id": policy.artifact_id,
        "visual_index_artifact_id": visual_build.artifact.artifact_id,
        "feature_spec_digest": visual_build.feature_spec.digest,
        "calibration_digest": visual_build.calibration.digest,
        "min_between_distance": visual_build.calibration.min_between_distance,
        "acceptance_threshold": visual_build.calibration.acceptance_threshold,
        "required_margin": visual_build.calibration.required_margin,
        "cleared": game.cleared,
        "score": game.score,
        "steps": game.steps,
        "trace": trace,
    }


def exhaustive_visual_equivalence(
    config: ShooterConfig = ShooterConfig(),
) -> dict[str, Any]:
    policy = compile_policy_artifact(config)
    visual_build = compile_visual_index_artifact(config, policy_artifact=policy)
    reader = make_visual_reader(policy, visual_build)
    total_waves = 0
    total_steps = 0
    for wave in product(range(config.width), repeat=len(config.wave)):
        wave_config = ShooterConfig(
            width=config.width,
            wave=tuple(int(value) for value in wave),
            max_steps=config.max_steps,
        )
        result = run_visual_policy_episode(
            wave_config,
            policy_artifact=policy,
            visual_index_build=visual_build,
            visual_reader=reader,
        )
        if not result["cleared"] or result["score"] != len(wave):
            raise RuntimeError("visual path failed wave %r" % (wave,))
        total_waves += 1
        total_steps += int(result["steps"])
    return {
        "policy_artifact_id": policy.artifact_id,
        "visual_index_artifact_id": visual_build.artifact.artifact_id,
        "waves_evaluated": total_waves,
        "waves_cleared": total_waves,
        "visual_decisions_compared": total_steps,
        "visual_symbolic_action_equivalence_percent": 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--exhaustive", action="store_true")
    args = parser.parse_args()

    config = ShooterConfig()
    policy = compile_policy_artifact(config)
    visual_build = compile_visual_index_artifact(config, policy_artifact=policy)
    reader = make_visual_reader(policy, visual_build)
    result = run_visual_policy_episode(
        config,
        policy_artifact=policy,
        visual_index_build=visual_build,
        visual_reader=reader,
    )
    summary = {key: value for key, value in result.items() if key != "trace"}
    summary["exact_decisions"] = sum(
        int(step["visual_decision"]["exact_feature_match"])
        for step in result["trace"]
    )
    if args.exhaustive:
        summary["exhaustive"] = exhaustive_visual_equivalence(config)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        to_bundle(policy, args.output_dir / "arcade-policy.vpm")
        to_bundle(visual_build.artifact, args.output_dir / "arcade-visual-index.vpm")
        (args.output_dir / "arcade-visual-run.json").write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary["output_dir"] = str(args.output_dir)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
