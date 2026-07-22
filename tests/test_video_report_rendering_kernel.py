import ast
import inspect

from zeromodel.video.domains.video_action_set import report_rendering


PROFILES = [
    {"provider_id": "P1", "mean_seconds_per_frame": 0.125, "frame_count": 2},
    {"provider_id": "P2", "mean_seconds_per_frame": 1.5, "frame_count": 2},
    {"provider_id": "P3", "mean_seconds_per_frame": 12.25, "frame_count": 2},
]


def test_static_reports_match_complete_pre_extraction_text() -> None:
    assert (
        report_rendering.benchmark_readme()
        == "# Video Action-Set Reachability Benchmark v1\n\nThis directory contains the frozen contract identities and the materialized development, calibration, and selection benchmark evidence.\n"
    )
    assert (
        report_rendering.reproduction_instructions()
        == "Run the benchmark CLI with `--freeze-benchmark`, `--build-development`, `--build-calibration`, `--build-selection`,\n`--audit-evidence-completeness`, `--audit-canonical-providers`, and `--verify-prospective-instrument`.\n"
    )


def test_runtime_reports_match_complete_pre_extraction_text() -> None:
    expected_reference = "# Runtime Profile Reference\n\n- P1: 0.125000s/frame over 2 frames\n- P2: 1.500000s/frame over 2 frames\n- P3: 12.250000s/frame over 2 frames\n"
    expected_optimized = expected_reference.replace("Reference", "Optimized", 1)
    assert report_rendering.runtime_profile_reference(PROFILES) == expected_reference
    assert report_rendering.runtime_profile_optimized(PROFILES) == expected_optimized


def test_report_renderer_has_no_filesystem_or_scientific_imports() -> None:
    tree = ast.parse(inspect.getsource(report_rendering))
    imports = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imports.isdisjoint({"pathlib", "os", "json", "zeromodel"})
