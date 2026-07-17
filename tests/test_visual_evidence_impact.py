from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_guard_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_visual_evidence_impact.py"
    spec = importlib.util.spec_from_file_location("check_visual_evidence_impact_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_guard = _load_guard_module()
CLAIMS_AUDIT_PATH = _guard.CLAIMS_AUDIT_PATH
EXEMPTION_PATH = _guard.EXEMPTION_PATH
evaluate_changed_files = _guard.evaluate_changed_files


def test_guard_passes_when_no_guarded_files_change() -> None:
    result = evaluate_changed_files(["README.md"])
    assert result.requirement_satisfied is True
    assert result.guarded_changed_files == ()


def test_guard_requires_evidence_change_for_guarded_files() -> None:
    result = evaluate_changed_files(["zeromodel/visual_analysis.py"])
    assert result.requirement_satisfied is False
    assert result.guarded_changed_files == ("zeromodel/visual_analysis.py",)
    assert CLAIMS_AUDIT_PATH in result.failure_message()
    assert EXEMPTION_PATH in result.failure_message()


def test_guard_accepts_claims_audit_change() -> None:
    result = evaluate_changed_files(
        ["zeromodel/visual_system_b.py", "docs/claims-audit.md"]
    )
    assert result.requirement_satisfied is True


def test_guard_accepts_research_protocol_change() -> None:
    result = evaluate_changed_files(
        [
            "examples/arcade_visual_system_b_adjudication.py",
            "docs/research/visual-address-system-b-v2-adjudication.md",
        ]
    )
    assert result.requirement_satisfied is True


def test_guard_accepts_results_evidence_change() -> None:
    result = evaluate_changed_files(
        [
            "examples/arcade_visual_system_b_adjudication.py",
            "docs/results/visual-address-system-b-v2/final-summary.json",
        ]
    )
    assert result.requirement_satisfied is True


def test_guard_accepts_machine_readable_exemption() -> None:
    result = evaluate_changed_files(
        ["zeromodel/visual_benchmark.py", "docs/results/visual-evidence-impact-exemption.json"]
    )
    assert result.requirement_satisfied is True
