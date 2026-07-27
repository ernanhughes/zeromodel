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
    assert by_case[("arcade", "alien_target_identity")]["status"] == "insufficient_observability"
    assert by_case[("warehouse", "crate_identity")]["status"] == "compiled"

    # Manual-baseline wiring: every case except alien_target_identity has a
    # literal historical representation to compare against, evaluated on the
    # same held-out split as every other strategy.
    for (domain, case_name), r in by_case.items():
        manual = r["manual_strategy"]
        if (domain, case_name) == ("arcade", "alien_target_identity"):
            assert manual["candidate"] is None
            assert manual["note"] is not None
        else:
            assert manual["candidate"] is not None, f"{domain}/{case_name} missing a manual reference"
            assert manual["held_out_eval"] is not None
