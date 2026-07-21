"""Canonical filesystem layout for the video action-set instrument."""

from __future__ import annotations

from pathlib import Path


ROOT_ARTIFACT_FILENAMES = (
    "benchmark-contract-identity.json",
    "generator-identity.json",
    "benchmark-manifest.json",
    "policy-artifact.json",
    "reachability-tile-reference.json",
    "episode-family-registry.json",
    "transformation-family-contract.json",
    "provider-manifest.json",
    "provider-formulas.json",
    "score-quantizer.json",
    "region-manifest.json",
    "split-manifest.json",
    "episode-plan.json",
    "final-split-sealed-plan.json",
    "final-split-sealed-digest.json",
    "evidence-schema.json",
    "phase-access-audits.json",
    "README.md",
    "reproduction.md",
    "runtime-profile-reference.json",
    "runtime-profile-optimized.json",
    "runtime-comparison.json",
    "runtime-profile-reference.md",
    "runtime-profile-optimized.md",
    "provider-runtime-equivalence.json",
    "provider-runtime-equivalence.csv",
    "observation-identity-manifest.json",
    "split-overlap-audit.json",
    "evidence-completeness-summary.json",
    "canonical-provider-results.csv",
    "canonical-provider-summary.json",
    "provider-equivalence-results.json",
    "tie-safety-results.json",
    "reference-closure-report.json",
)

SPLIT_ARTIFACT_FILENAMES = (
    "frame-metadata.jsonl",
    "provider-evidence.jsonl",
)

SPLITS = ("development", "calibration", "selection", "final")
MATERIALIZED_SPLITS = SPLITS[:-1]


def root_artifact_path(output_dir: Path, filename: str) -> Path:
    return output_dir / filename


def split_artifact_path(output_dir: Path, split: str, filename: str) -> Path:
    return output_dir / split / filename


def split_manifest_path(output_dir: Path, split: str) -> Path:
    return output_dir / f"{split}-manifest.json"


def family_closure_path(output_dir: Path, split: str) -> Path:
    return output_dir / f"{split}-family-closure-report.json"


def benchmark_database_path(output_dir: Path) -> Path:
    return output_dir / "benchmark.sqlite3"


def reachability_tile_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "docs"
        / "results"
        / "video-policy-reachability-tile-v1"
        / "reachability-tile.json"
    )


def canonical_relative_paths() -> tuple[str, ...]:
    """Return recognized canonical paths, not a produced-artifact manifest.

    The registry includes optional root outputs and evidence for materialized splits.
    Dynamic split manifests, family-closure reports, and the SQLite database are
    addressed by their path helpers instead of enumerated here.
    """
    split_paths = tuple(
        f"{split}/{filename}"
        for split in MATERIALIZED_SPLITS
        for filename in SPLIT_ARTIFACT_FILENAMES
    )
    return ROOT_ARTIFACT_FILENAMES + split_paths
