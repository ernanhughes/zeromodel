"""Pure human-readable projections for video action-set artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def benchmark_readme() -> str:
    return (
        "# Video Action-Set Reachability Benchmark v1\n\n"
        "This directory contains the frozen contract identities and the "
        "materialized development, calibration, and selection benchmark evidence.\n"
    )


def reproduction_instructions() -> str:
    return (
        "Run the benchmark CLI with `--freeze-benchmark`, `--build-development`, "
        "`--build-calibration`, `--build-selection`,\n"
        "`--audit-evidence-completeness`, `--audit-canonical-providers`, and "
        "`--verify-prospective-instrument`.\n"
    )


def runtime_profile_reference(profiles: Sequence[Mapping[str, Any]]) -> str:
    return _runtime_profile("Runtime Profile Reference", profiles)


def runtime_profile_optimized(profiles: Sequence[Mapping[str, Any]]) -> str:
    return _runtime_profile("Runtime Profile Optimized", profiles)


def _runtime_profile(title: str, profiles: Sequence[Mapping[str, Any]]) -> str:
    return (
        "\n".join(
            [f"# {title}", ""]
            + [
                f"- {item['provider_id']}: "
                f"{item['mean_seconds_per_frame']:.6f}s/frame over "
                f"{item['frame_count']} frames"
                for item in profiles
            ]
        )
        + "\n"
    )
