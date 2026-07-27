import json
from pathlib import Path

from visual_transition_benchmark import compiler_run as cr


def test_compiler_run_end_to_end_small(tmp_path: Path):
    exit_code = cr.main(
        [
            "--dev-samples",
            "2",
            "--eval-samples",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "compiler-results.json").exists()
    assert (tmp_path / "compiler-summary.md").exists()

    payload = json.loads((tmp_path / "compiler-results.json").read_text(encoding="utf-8"))
    assert payload["environment"]["case_count"] == 12
    results = payload["results"]
    assert len(results) == 12

    statuses = {r["status"] for r in results}
    assert statuses <= {"compiled", "insufficient_representation", "insufficient_observability"}

    by_case = {(r["domain"], r["case"]): r for r in results}
    assert by_case[("arcade", "alien_target_identity")]["status"] != "compiled"
    assert by_case[("warehouse", "crate_identity")]["status"] == "compiled"
