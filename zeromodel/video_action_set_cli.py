"""Command-line dispatch for the video action-set scientific instrument."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .domains.video_action_set.build_orchestration import (
    build_split,
    freeze_benchmark,
    profile_runtime,
)
from .domains.video_action_set.provider_measurement import SplitBuildProgress
from .domains.video_action_set.mutation_orchestration import verify_instrument
from .domains.video_action_set.verification_orchestration import (
    audit_canonical_providers,
    audit_evidence_completeness,
    verify_provider_runtime_equivalence,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    REPO_ROOT / "docs" / "results" / "video-action-set-reachability-benchmark-v1"
)
_PROGRESS_FRAME_CADENCE = 25


def _stderr_progress_observer():
    last_reported = 0

    def observe(event: SplitBuildProgress) -> None:
        nonlocal last_reported
        is_final = event.processed_frame_count == event.total_frame_count
        if (
            not is_final
            and event.processed_frame_count - last_reported < _PROGRESS_FRAME_CADENCE
        ):
            return
        last_reported = event.processed_frame_count
        print(
            (
                f"{event.split}: {event.processed_frame_count}/"
                f"{event.total_frame_count} frames "
                f"({event.percentage_complete:.1f}%), "
                f"scoreable={event.scoreable_frame_count_processed}, "
                f"typed_gaps={event.typed_gap_count_processed}, "
                f"provider_calls={event.provider_scoring_calls_completed}, "
                f"elapsed={event.elapsed_seconds:.1f}s"
            ),
            file=sys.stderr,
        )

    return observe


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--freeze-benchmark", action="store_true")
    parser.add_argument("--build-development", action="store_true")
    parser.add_argument("--build-calibration", action="store_true")
    parser.add_argument("--build-selection", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--audit-evidence-completeness", action="store_true")
    parser.add_argument("--audit-canonical-providers", action="store_true")
    parser.add_argument("--profile-runtime", action="store_true")
    parser.add_argument("--verify-provider-runtime-equivalence", action="store_true")
    parser.add_argument(
        "--profile-provider",
        choices=("P1", "P2", "P3", "all"),
        default="all",
    )
    parser.add_argument("--profile-frame-count", type=int, default=8)
    parser.add_argument("--verify-prospective-instrument", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    progress_observer = _stderr_progress_observer() if args.progress else None
    selected = sum(
        int(flag)
        for flag in (
            args.freeze_benchmark,
            args.build_development,
            args.build_calibration,
            args.build_selection,
            args.audit_evidence_completeness,
            args.audit_canonical_providers,
            args.profile_runtime,
            args.verify_provider_runtime_equivalence,
            args.verify_prospective_instrument,
        )
    )
    if selected != 1:
        raise SystemExit("exactly one command is required")
    if args.freeze_benchmark:
        payload = freeze_benchmark(args.output_dir, REPO_ROOT)
    elif args.build_development:
        freeze_benchmark(args.output_dir, REPO_ROOT)
        payload = build_split(
            "development",
            args.output_dir,
            REPO_ROOT,
            progress_observer=progress_observer,
        )
    elif args.build_calibration:
        freeze_benchmark(args.output_dir, REPO_ROOT)
        payload = build_split(
            "calibration",
            args.output_dir,
            REPO_ROOT,
            progress_observer=progress_observer,
        )
    elif args.build_selection:
        freeze_benchmark(args.output_dir, REPO_ROOT)
        payload = build_split(
            "selection",
            args.output_dir,
            REPO_ROOT,
            progress_observer=progress_observer,
        )
    elif args.audit_evidence_completeness:
        payload = audit_evidence_completeness(args.output_dir)
    elif args.audit_canonical_providers:
        payload = audit_canonical_providers(args.output_dir)
    elif args.profile_runtime:
        payload = profile_runtime(
            args.output_dir,
            REPO_ROOT,
            provider=args.profile_provider,
            frame_count=args.profile_frame_count,
        )
    elif args.verify_provider_runtime_equivalence:
        payload = verify_provider_runtime_equivalence(args.output_dir, REPO_ROOT)
    else:
        payload = verify_instrument(args.output_dir, REPO_ROOT)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
