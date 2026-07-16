from __future__ import annotations

import importlib.util
from pathlib import Path

from zeromodel import VPMPolicyLookup


def _load_demo():
    path = Path(__file__).resolve().parents[1] / "examples" / "arcade_shooter_policy.py"
    spec = importlib.util.spec_from_file_location("arcade_shooter_policy", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arcade_policy_compiles_to_vpm_and_reads_signs() -> None:
    demo = _load_demo()
    artifact = demo.compile_policy_artifact()
    reader = VPMPolicyLookup(artifact)

    assert artifact.source.metadata["slogan"] == "signs_not_directions"
    assert reader.read(demo.state_row_id(3, 0, 0)).action == "LEFT"
    assert reader.read(demo.state_row_id(0, 0, 0)).action == "FIRE"
    assert reader.read(demo.state_row_id(0, 6, 1)).action == "RIGHT"


def test_arcade_policy_clears_wave_and_beats_random_baseline() -> None:
    demo = _load_demo()
    config = demo.ShooterConfig()

    policy = demo.run_policy_episode(config)
    random_average = demo.random_baseline_average(config, seeds=10)

    assert policy["cleared"] is True
    assert policy["score"] == len(config.wave)
    assert policy["score"] >= random_average + 1.5


def test_same_artifact_and_seed_produce_identical_action_trace() -> None:
    demo = _load_demo()
    config = demo.ShooterConfig()

    first = demo.run_policy_episode(config)
    second = demo.run_policy_episode(config)

    assert first["artifact_id"] == second["artifact_id"]
    assert [step["action"] for step in first["trace"]] == [step["action"] for step in second["trace"]]
    assert [(step["row_id"], step["source_metric_index"]) for step in first["trace"]] == [
        (step["row_id"], step["source_metric_index"]) for step in second["trace"]
    ]
