"""Fail closed when high-impact visual changes lack matching evidence records.

This checker is intentionally local and network-free. Repository automation can
enforce it on pull requests, but branch protection must separately require pull
requests and passing checks; a workflow alone cannot prevent authorized direct
pushes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


CLAIMS_AUDIT_PATH = "docs/claims-audit.md"
EXEMPTION_PATH = "docs/results/visual-evidence-impact-exemption.json"
GUARDED_IMPLEMENTATION_PATHS = (
    "zeromodel/visual_dataset.py",
    "zeromodel/visual_experiment.py",
    "zeromodel/visual_benchmark.py",
    "zeromodel/visual_analysis.py",
    "zeromodel/visual_system_b.py",
    "examples/arcade_visual_address_benchmark.py",
    "examples/arcade_visual_system_b_adjudication.py",
)
RELEVANT_RESEARCH_PREFIXES = (
    "docs/research/visual-",
    "docs/research/fixed-camera-",
)
RELEVANT_RESULTS_PREFIX = "docs/results/visual-address-"


@dataclass(frozen=True)
class EvidenceImpactResult:
    changed_files: tuple[str, ...]
    guarded_changed_files: tuple[str, ...]
    requirement_satisfied: bool
    reasons: tuple[str, ...]

    def failure_message(self) -> str:
        lines = ["Visual evidence-impact guard failed."]
        if self.guarded_changed_files:
            lines.append("Guarded changed files:")
            for path in self.guarded_changed_files:
                lines.append(f" - {path}")
        if self.reasons:
            lines.append("Missing requirement:")
            for reason in self.reasons:
                lines.append(f" - {reason}")
        return "\n".join(lines)


def _normalize(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({path.replace("\\", "/") for path in paths if path.strip()}))


def _git_changed_files(*, base: str, head: str, repo_root: Path) -> tuple[str, ...]:
    output = subprocess.check_output(
        ["git", "diff", "--name-only", base, head],
        cwd=str(repo_root),
        text=True,
    )
    return _normalize(output.splitlines())


def evaluate_changed_files(changed_files: Sequence[str]) -> EvidenceImpactResult:
    changed = _normalize(changed_files)
    guarded = tuple(path for path in changed if path in GUARDED_IMPLEMENTATION_PATHS)
    if not guarded:
        return EvidenceImpactResult(
            changed_files=changed,
            guarded_changed_files=(),
            requirement_satisfied=True,
            reasons=(),
        )

    claims_changed = CLAIMS_AUDIT_PATH in changed
    research_changed = any(path.startswith(RELEVANT_RESEARCH_PREFIXES) for path in changed)
    results_changed = any(path.startswith(RELEVANT_RESULTS_PREFIX) for path in changed)
    exemption_changed = EXEMPTION_PATH in changed

    if claims_changed or research_changed or results_changed or exemption_changed:
        return EvidenceImpactResult(
            changed_files=changed,
            guarded_changed_files=guarded,
            requirement_satisfied=True,
            reasons=(),
        )

    return EvidenceImpactResult(
        changed_files=changed,
        guarded_changed_files=guarded,
        requirement_satisfied=False,
        reasons=(
            "change docs/claims-audit.md, a relevant docs/research visual protocol/adjudication file, "
            "a relevant docs/results/visual-address-* evidence record, or "
            f"{EXEMPTION_PATH}",
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        help="Explicit changed file path. May be provided multiple times.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args(argv)

    if args.changed_file:
        changed = _normalize(args.changed_file)
    elif args.base and args.head:
        changed = _git_changed_files(base=args.base, head=args.head, repo_root=args.repo_root)
    else:
        parser.error("provide either --changed-file entries or both --base and --head")

    result = evaluate_changed_files(changed)
    print("Changed files:")
    for path in result.changed_files:
        print(f" - {path}")

    if result.requirement_satisfied:
        if result.guarded_changed_files:
            print("Visual evidence-impact requirement satisfied.")
        else:
            print("No guarded implementation files changed.")
        return 0

    print(result.failure_message(), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
