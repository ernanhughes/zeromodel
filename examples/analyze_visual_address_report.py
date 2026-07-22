"""Generate an exploratory failure atlas from a traced visual benchmark report.

This does not select a deployment threshold. It reuses evaluation traces and is
therefore diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.visual.visual_analysis import analyze_trace_sets  # noqa: E402


def _trace_sets(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    artifacts = payload.get("artifacts") or {}
    result: Dict[str, Any] = {}
    for key, value in artifacts.items():
        if str(key).startswith("traces_"):
            result[str(key)[len("traces_"):]] = value
    if not result:
        raise ValueError(
            "report contains no traces; rerun with --include-traces"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    atlas = analyze_trace_sets(_trace_sets(payload))
    text = json.dumps(atlas, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
