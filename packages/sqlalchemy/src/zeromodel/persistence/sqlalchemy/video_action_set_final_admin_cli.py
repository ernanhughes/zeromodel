"""Read-only final-access ledger inspection commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zeromodel.persistence.sqlalchemy.db.runtime import (
    build_finalization_sqlite_runtime,
)
from zeromodel.persistence.sqlalchemy.db.session import sqlite_database_url
from zeromodel.video.domains.video_action_set.final_access_dto import (
    validate_final_identifier,
)
from zeromodel.video.domains.video_action_set.final_reconstruction import (
    reconstruct_final_access_ledger,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("observe", "reconstruct"):
        command = subparsers.add_parser(name)
        command.add_argument("--database-path", type=Path, required=True)
        command.add_argument("--access-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    validate_final_identifier(args.access_id, "final access id mismatch")
    database_url = sqlite_database_url(args.database_path)
    runtime = build_finalization_sqlite_runtime(database_url)
    payload = reconstruct_final_access_ledger(
        runtime.video_action_set.engine.final_access_service.store,
        args.access_id,
    )
    result = (
        {
            "access_id": args.access_id,
            "state": payload["record"]["state"],
            "publication_status": payload["publication_status"],
            "publishable_success": payload["publishable_success"],
            "counters": payload["counters"],
        }
        if args.command == "observe"
        else payload
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
