from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zeromodel.video_action_set_benchmark import (  # noqa: E402
    audit_canonical_providers,
    audit_evidence_completeness,
    build_split,
    freeze_benchmark,
    verify_instrument,
)


OUTPUT_DIR = REPO_ROOT / "docs" / "results" / "video-action-set-reachability-benchmark-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--freeze-benchmark", action="store_true")
    parser.add_argument("--build-development", action="store_true")
    parser.add_argument("--build-calibration", action="store_true")
    parser.add_argument("--build-selection", action="store_true")
    parser.add_argument("--audit-evidence-completeness", action="store_true")
    parser.add_argument("--audit-canonical-providers", action="store_true")
    parser.add_argument("--verify-prospective-instrument", action="store_true")
    args = parser.parse_args()
    selected = sum(
        int(flag)
        for flag in (
            args.freeze_benchmark,
            args.build_development,
            args.build_calibration,
            args.build_selection,
            args.audit_evidence_completeness,
            args.audit_canonical_providers,
            args.verify_prospective_instrument,
        )
    )
    if selected != 1:
        raise SystemExit("exactly one command is required")
    if args.freeze_benchmark:
        payload = freeze_benchmark(args.output_dir, REPO_ROOT)
    elif args.build_development:
        freeze_benchmark(args.output_dir, REPO_ROOT)
        payload = build_split("development", args.output_dir, REPO_ROOT)
    elif args.build_calibration:
        freeze_benchmark(args.output_dir, REPO_ROOT)
        payload = build_split("calibration", args.output_dir, REPO_ROOT)
    elif args.build_selection:
        freeze_benchmark(args.output_dir, REPO_ROOT)
        payload = build_split("selection", args.output_dir, REPO_ROOT)
    elif args.audit_evidence_completeness:
        payload = audit_evidence_completeness(args.output_dir)
    elif args.audit_canonical_providers:
        payload = audit_canonical_providers(args.output_dir)
    else:
        payload = verify_instrument(args.output_dir, REPO_ROOT)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
