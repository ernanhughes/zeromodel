from pathlib import Path

from visual_transition_benchmark import cross_domain_run as cdr


def test_cross_domain_run_end_to_end_small(tmp_path: Path):
    exit_code = cdr.main(
        [
            "--arcade-dev-episodes",
            "1",
            "--arcade-eval-episodes",
            "2",
            "--warehouse-dev-episodes",
            "1",
            "--warehouse-eval-episodes",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "cross-domain-results.json").exists()
    assert (tmp_path / "cross-domain-summary.md").exists()
    assert (tmp_path / "domain-results" / "arcade.json").exists()
    assert (tmp_path / "domain-results" / "warehouse.json").exists()
    assert (tmp_path / "transition-level-results.jsonl").exists()
    assert (tmp_path / "visual-index.html").exists()

    lines = (tmp_path / "transition-level-results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0

    import json

    results = json.loads((tmp_path / "cross-domain-results.json").read_text(encoding="utf-8"))
    assert "capability_table" in results
    assert results["domain_reports"]["arcade"]["n"] > 0
    assert results["domain_reports"]["warehouse"]["n"] > 0
