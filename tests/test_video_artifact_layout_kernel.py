from pathlib import Path

from zeromodel.domains.video_action_set import artifact_layout as layout


def test_canonical_artifact_layout_is_frozen() -> None:
    root = Path("benchmark")

    assert (
        layout.root_artifact_path(root, "episode-plan.json")
        == root / "episode-plan.json"
    )
    assert (
        layout.split_artifact_path(root, "selection", "provider-evidence.jsonl")
        == root / "selection" / "provider-evidence.jsonl"
    )
    assert (
        layout.split_manifest_path(root, "calibration")
        == root / "calibration-manifest.json"
    )
    assert (
        layout.family_closure_path(root, "development")
        == root / "development-family-closure-report.json"
    )
    assert layout.benchmark_database_path(root) == root / "benchmark.sqlite3"
    paths = layout.canonical_relative_paths()
    assert paths[:4] == (
        "benchmark-contract-identity.json",
        "generator-identity.json",
        "benchmark-manifest.json",
        "policy-artifact.json",
    )
    assert paths[-2:] == (
        "selection/frame-metadata.jsonl",
        "selection/provider-evidence.jsonl",
    )
    assert "final/frame-metadata.jsonl" not in paths
    assert "final/provider-evidence.jsonl" not in paths
    assert "final-split-sealed-plan.json" in paths
    assert "final-split-sealed-digest.json" in paths


def test_canonical_paths_are_a_registry_not_a_produced_manifest() -> None:
    assert "not a produced-artifact manifest" in (
        layout.canonical_relative_paths.__doc__ or ""
    )
    assert "development-manifest.json" not in layout.canonical_relative_paths()
    assert "benchmark.sqlite3" not in layout.canonical_relative_paths()


def test_reachability_tile_path_preserves_historical_layout() -> None:
    root = Path("repo")
    assert (
        layout.reachability_tile_path(root)
        == root
        / "docs"
        / "results"
        / "video-policy-reachability-tile-v1"
        / "reachability-tile.json"
    )
