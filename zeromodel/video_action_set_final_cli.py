"""Command-line entrypoint for final video action-set access orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .db.runtime import build_finalization_sqlite_runtime
from .domains.video_action_set.final_access_dto import (
    FINAL_EXECUTION_REQUEST_VERSION,
    FinalExecutionRequestDTO,
)
from .domains.video_action_set.final_access_service import (
    load_final_authorization_file,
)
from .runtime import build_runtime


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorization-file", type=Path, required=True)
    parser.add_argument("--expected-authorization-digest", required=True)
    parser.add_argument("--expected-sealed-plan-digest", required=True)
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--operator-identity", default="operator")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    authorization = load_final_authorization_file(args.authorization_file)
    request = FinalExecutionRequestDTO.create(
        {
            "version": FINAL_EXECUTION_REQUEST_VERSION,
            "output_dir": str(args.output_dir),
            "authorization_file": str(args.authorization_file),
            "expected_authorization_digest": args.expected_authorization_digest,
            "expected_sealed_plan_digest": args.expected_sealed_plan_digest,
            "database_path": str(args.database_path),
            "preflight_only": bool(args.preflight_only),
            "operator_identity": args.operator_identity,
            "unattended": bool(args.non_interactive),
            "request_payload": {
                "cli": "zeromodel.video_action_set_final_cli",
                "preflight_only": bool(args.preflight_only),
            },
        }
    )
    if args.preflight_only:
        runtime = build_runtime()
        payload = runtime.video_action_set.execute_final_once(request)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.non_interactive and not authorization.unattended_permitted:
        raise SystemExit("authorization does not permit unattended final execution")
    if not args.non_interactive:
        print(authorization.operator_confirmation_text, file=sys.stderr)
        confirmation = input("Type the confirmation text exactly: ")
        if confirmation != authorization.operator_confirmation_text:
            raise SystemExit("final execution confirmation mismatch")
    database_url = args.database_path.resolve().as_uri().replace("file:///", "sqlite:///")
    initialize_authority = (
        not args.database_path.exists() or args.database_path.stat().st_size == 0
    )
    runtime = build_finalization_sqlite_runtime(
        database_url,
        initialize_authority=initialize_authority,
    )
    payload = runtime.video_action_set.execute_final_once(request)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
