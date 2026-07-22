from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from zeromodel.core.policy_lookup import VPMPolicyLookup


def _load_demo():
    path = Path(__file__).resolve().parents[1] / "examples" / "criticality_verification.py"
    spec = importlib.util.spec_from_file_location("criticality_verification", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_q_policy_preserves_actions_and_adds_diagnostics() -> None:
    demo = _load_demo()
    artifact = demo.compile_q_policy_artifact()
    reader = VPMPolicyLookup(
        artifact,
        action_metric_ids=demo.ACTIONS,
        evidence_metric_ids=demo.EVIDENCE_METRICS,
    )

    checked = 0
    targets = (None,) + tuple(range(7))
    for tank_x in range(7):
        for target_x in targets:
            for cooldown in (0, 1):
                values = demo.teacher_q_values(tank_x, target_x, cooldown)
                expected_index = max(
                    range(len(demo.ACTIONS)),
                    key=lambda index: (values[index], -index),
                )
                decision = reader.read(
                    demo.state_row_id(tank_x, target_x, cooldown)
                )
                assert decision.action == demo.ACTIONS[expected_index]
                assert set(decision.evidence) == set(demo.EVIDENCE_METRICS)
                checked += 1

    assert checked == 112


def test_counterexample_repair_and_verification_lineage(tmp_path) -> None:
    demo = _load_demo()
    result = demo.run_demo(tmp_path)

    assert result["original_passed"] is True
    assert result["unsafe_passed"] is False
    assert result["repaired_passed"] is True
    assert len(result["counterexamples"]) == 1
    assert result["counterexamples"][0]["row_id"] == "tank=0|target=1|cooldown=0"
    assert result["counterexamples"][0]["action"] == "FIRE"
    assert result["original_policy_id"] != result["unsafe_policy_id"]
    assert result["unsafe_policy_id"] != result["repaired_policy_id"]
    assert result["failed_verification_id"] != result["repaired_verification_id"]
    assert (tmp_path / "criticality_verification_results.json").exists()
    assert (tmp_path / "verification_failed.vpm").exists()
    assert (tmp_path / "verification_repaired.vpm").exists()
