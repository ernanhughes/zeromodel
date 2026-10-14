"""ZM-HAND-01 command line.

    python -m examples.hand_anatomy.cli --help

Exit codes: 0 success; 1 validation/replay mismatch; 2 unsupported or invalid input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

from . import _bootstrap  # noqa: F401
from ._common import HandAnatomyError, UnsupportedInput
from .assessment import DecisionContract
from .pose_library import (
    TemplateLibrary,
    build_library,
    load_library,
    save_library,
    validate_library,
)

DEFAULT_OUT = "artifacts/hand_anatomy"


def _load_contract(path: Optional[str]) -> DecisionContract:
    if not path:
        return DecisionContract()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DecisionContract.from_dict(data.get("contract", data))


def _library(path: Optional[str]) -> TemplateLibrary:
    return load_library(path) if path else build_library()


def cmd_build_library(args: argparse.Namespace) -> int:
    lib = build_library(thumb_variants=tuple(args.thumb_variants))
    problems = validate_library(lib)
    if problems:
        print("library has anatomical inconsistencies:\n  " + "\n  ".join(problems[:10]))
        return 1
    save_library(lib, args.output)
    from .visualization import contact_sheet

    contact_sheet(lib, Path(args.output) / "contact_sheet_right.png", "right")
    contact_sheet(lib, Path(args.output) / "contact_sheet_left.png", "left")
    print("templates=%d library_id=%s -> %s" % (len(lib), lib.library_id, args.output))
    return 0


def cmd_inspect_library(args: argparse.Namespace) -> int:
    lib = load_library(args.library)
    meta = lib.metadata()
    by_hand: Dict[str, int] = {}
    for t in lib.templates:
        by_hand[t.pose.handedness] = by_hand.get(t.pose.handedness, 0) + 1
    print(json.dumps({k: meta[k] for k in ("library_id", "geometry_version", "topology_id",
                                           "render_recipe_id", "size")}, indent=1))
    print("by_handedness=%s  consistency_problems=%d" % (by_hand, len(validate_library(lib))))
    return 0


def cmd_verify_library(args: argparse.Namespace) -> int:
    saved = load_library(args.library)  # re-renders and verifies every template
    fresh = build_library()
    ok = saved.library_id == fresh.library_id
    print("regenerated library identical: %s (%s)" % (ok, fresh.library_id))
    return 0 if ok else 1


def cmd_contact_sheet(args: argparse.Namespace) -> int:
    from .visualization import contact_sheet

    lib = _library(args.library)
    print(contact_sheet(lib, args.output, args.hand))
    return 0


def cmd_make_fixtures(args: argparse.Namespace) -> int:
    from .synthetic import write_demo_fixtures

    for name, path in write_demo_fixtures(args.output).items():
        print("%s: %s" % (name, path))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from .evidence import inspect_files

    lib = _library(args.library)
    report = inspect_files(
        lib, image=args.image, mask=args.mask, landmarks=args.landmarks,
        segment_background=args.segment_background, contract=_load_contract(args.contract),
        out_dir=args.output,
    )
    if report["decision"] == "unsupported_input":
        print("decision=unsupported_input reason=%s" % report["decision_detail"]["reasons"][0])
        return 2
    ev = report["evidence"]["best_template"]
    print(
        "decision=%s best=%s IoU=%.3f explainer=%s protrusion_expl=%.3f observed_expl=%.3f calibrated=%s"
        % (report["decision"], report["best_template_name"], ev["silhouette_agreement"],
           report["explainer_template_name"], report["evidence"]["explainer_template"]["protrusion_explainability"],
           report["evidence"]["explainer_template"]["observed_explainability"],
           report["decision_contract"]["calibrated"])
    )
    print("reasons=%s vpm_artifact_id=%s" % (report["decision_detail"]["reasons"], report["vpm_artifact_id"]))
    print("report: %s" % (Path(args.output) / "report.json"))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark

    final = run_benchmark(args.output, workers=args.workers)
    meta = final["meta"]
    print("cases=%s leakage_ok=%s wall_s=%s" % (meta["split_sizes"], meta["leakage_checks"]["ok"], meta["runtime"]["total_wall_s"]))
    for name, res in final["results"].items():
        s = res["test"]["all"]
        print("%-48s tp=%d fp=%d tn=%d fn=%d abst=%d recall=%s fpr=%s" % (
            name, s["tp"], s["fp"], s["tn"], s["fn"], s["abstained_malformed"] + s["abstained_valid"],
            s["recall_overall"], s["fpr_decided"]))
    print("results: %s" % (Path(args.output) / "results.md"))
    return 0


def cmd_evaluate_external(args: argparse.Namespace) -> int:
    from .external import evaluate_external

    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    out = evaluate_external(args.root, _library(args.library), DecisionContract.from_dict(cal["contract"]),
                            float(cal["baseline_iou_threshold"]), args.output)
    for name in ("structured_evidence", "iou_baseline"):
        s = out[name]
        print("%-20s tp=%d fp=%d tn=%d fn=%d abstained=%d" % (
            name, s["tp"], s["fp"], s["tn"], s["fn"], s["abstained_malformed"] + s["abstained_valid"]))
    return 0


def cmd_validate_report(args: argparse.Namespace) -> int:
    from .evidence import validate_report

    problems = validate_report(json.loads(Path(args.report).read_text(encoding="utf-8")))
    print("valid" if not problems else "INVALID:\n  " + "\n  ".join(problems))
    return 0 if not problems else 1


def cmd_replay(args: argparse.Namespace) -> int:
    from .evidence import replay_report
    from .observations import load_observation

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    obs = load_observation(args.image, args.mask, args.landmarks, args.segment_background)
    problems = replay_report(report, obs, _library(args.library))
    print("replay reproduced the report" if not problems else "MISMATCH:\n  " + "\n  ".join(problems))
    return 0 if not problems else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m examples.hand_anatomy.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("build-library", help="generate, validate and save the template library")
    s.add_argument("--output", default=DEFAULT_OUT + "/library")
    s.add_argument("--thumb-variants", nargs="+", default=["base"], choices=["base", "out", "in"])
    s.set_defaults(fn=cmd_build_library)

    s = sub.add_parser("inspect-library", help="print library metadata")
    s.add_argument("--library", required=True)
    s.set_defaults(fn=cmd_inspect_library)

    s = sub.add_parser("verify-library", help="regenerate the library and compare identities")
    s.add_argument("--library", required=True)
    s.set_defaults(fn=cmd_verify_library)

    s = sub.add_parser("contact-sheet", help="render a template contact sheet")
    s.add_argument("--library")
    s.add_argument("--hand", choices=["left", "right"])
    s.add_argument("--output", default=DEFAULT_OUT + "/contact_sheet.png")
    s.set_defaults(fn=cmd_contact_sheet)

    s = sub.add_parser("make-fixtures", help="write demo masks (plausible + six-finger) from the independent generator")
    s.add_argument("--output", default=DEFAULT_OUT + "/fixtures")
    s.set_defaults(fn=cmd_make_fixtures)

    def inputs(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--image", help="RGB/RGBA image (PNG/JPEG/BMP)")
        sp.add_argument("--mask", help="binary hand mask (white = hand); same size as --image if both given")
        sp.add_argument("--landmarks", help="optional JSON with 21 (x, y) landmarks from an external provider")
        sp.add_argument("--segment-background", action="store_true",
                        help="bounded fallback: near-uniform background threshold (declares its assumption)")
        sp.add_argument("--library", help="saved library dir (default: build in memory)")

    s = sub.add_parser("inspect", help="inspect an image/mask against the template library")
    inputs(s)
    s.add_argument("--contract", help="decision-contract JSON (e.g. calibrated_contract.json from a benchmark run)")
    s.add_argument("--output", required=True)
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("benchmark", help="run the baseline-vs-structured benchmark")
    s.add_argument("--output", default=DEFAULT_OUT + "/benchmark")
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("evaluate-external", help="score annotated external crops (see fixtures/README.md)")
    s.add_argument("--root", required=True)
    s.add_argument("--calibration", required=True, help="calibrated_contract.json from a benchmark run")
    s.add_argument("--library")
    s.add_argument("--output", required=True)
    s.set_defaults(fn=cmd_evaluate_external)

    s = sub.add_parser("validate-report", help="self-consistency check of a saved report")
    s.add_argument("--report", required=True)
    s.set_defaults(fn=cmd_validate_report)

    s = sub.add_parser("replay", help="recompute a saved report from its original inputs")
    s.add_argument("--report", required=True)
    inputs(s)
    s.set_defaults(fn=cmd_replay)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (UnsupportedInput, HandAnatomyError, FileNotFoundError, json.JSONDecodeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
