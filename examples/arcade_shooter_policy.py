"""Tiny arcade-shooter policy demo for ZeroModel 1.0.

This is deliberately not a neural gameplay agent.  It compiles a closed-world
state/action policy into one VPM artifact, then the runtime reads the current
state row from that artifact and executes the action sign it finds there.

Run:

    python examples/arcade_shooter_policy.py
"""
from __future__ import annotations

import json
from zeromodel.arcade_policy.model import (
    ACTIONS,
    ShooterConfig,
    TinyArcadeShooter,
    compile_policy_artifact,
    random_baseline_average,
    run_policy_episode,
    run_random_episode,
    state_row_id,
)


if __name__ == "__main__":
    config = ShooterConfig()
    policy = run_policy_episode(config)
    baseline = random_baseline_average(config)
    print(json.dumps({
        "artifact_id": policy["artifact_id"],
        "policy_score": policy["score"],
        "policy_cleared": policy["cleared"],
        "policy_steps": policy["steps"],
        "random_average_score_10_seeds": baseline,
        "first_moves": policy["trace"][:8],
    }, indent=2, sort_keys=True))
