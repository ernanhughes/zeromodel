"""ZM-HAND-02 command line.

    python -m examples.hand_anatomy_02.cli --help

Exit codes: 0 success; 1 validation/replay mismatch; 2 unsupported or invalid input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

from examples.hand_anatomy import _bootstrap  # noqa: F401
from examples.hand_anatomy import visualization as vz
from examples.hand_anatomy._common import HandAnatomyError, UnsupportedInput
from examples.hand_anatomy.observations import load_observation, observation_from_arrays
from examples.hand_anatomy.pose_library import TemplateLibrary, build_library, load_library

from .vpm02 import Contract02

OUT = "artifacts/hand_anatomy_02"


def _library(path: Optional[str]) -> TemplateLibrary:
    return load_library(path) if path else build_library()


def _contract(path: Optional[str]) -> Contract02:
    if not path:
        return Contract02()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Contract02.from_dict(data.get("contract", data))


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark02 import run_benchmark02

    res = run_benchmark02(args.output, workers=args.workers, final=args.final, reuse_records=args.reuse_records)
    if not res.get("final"):
        print("calibration split scored (%d cases); pass --final to score the test split ONCE" % res["cal_cases"])
        return 0
    print("verdict C_residual: %s" % res["verdict_C_residual"])
    for m, v in res["evaluation"]["methods"].items():
        print("%-22s test_fpr=%.3f primary %d/%d" % (m, v["test_fpr"], v["primary"]["detected"], v["primary"]["pairs"]))
    print("results: %s" % (Path(args.output) / "results.md"))
    return 0


def cmd_pair_sheet(args: argparse.Namespace) -> int:
    from PIL import Image

    from .inspect02 import inspect02, residual_overlay
    from .pairs import PRIMARY_SUBTYPES, build_dataset

    lib = _library(args.library)
    contract = _contract(args.contract)
    pairs = [p for p in build_dataset()["test02"]["pairs"]
             if p.category == "extra_finger" and p.subtype in PRIMARY_SUBTYPES and not p.indistinguishable]
    tiles = []
    for p in pairs[:: max(1, len(pairs) // args.count)][: args.count]:
        for tag, mask in (("valid", p.valid), ("defect", p.defect)):
            res = inspect02(observation_from_arrays(mask), lib, contract)
            tiles.append(("%s %s C=%.2f A=%.2f %s" % (p.subtype[:4], tag, res.scores["C_residual"],
                                                       res.scores["A_iou"], res.decision["status"][:4]),
                          residual_overlay(res, 1)))
    sheet = vz.tile_sheet([(n, t if isinstance(t, Image.Image) else t) for n, t in tiles], columns=4, tile=190)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from .inspect02 import build_report02, inspect02, write_inspection02

    obs = load_observation(args.image, args.mask)
    res = inspect02(obs, _library(args.library), _contract(args.contract))
    write_inspection02(res, args.output)
    rep = build_report02(res)
    top = rep["evidence"]["diagnostics_explainer"]["top_residual"]
    print("decision=%s C_residual=%.3f A_iou=%.3f A_refined_iou=%.3f calibrated=%s" % (
        rep["decision"], rep["scores"]["C_residual"], rep["scores"]["A_iou"], rep["scores"]["A_refined_iou"],
        rep["decision_contract"]["calibrated"]))
    print("reasons=%s top_residual=%s" % (rep["decision_detail"]["reasons"], top and
          {k: top[k] for k in ("length", "width", "attached_to", "finger_likeness")}))
    print("report: %s" % (Path(args.output) / "report.json"))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from .inspect02 import validate_report02

    problems = validate_report02(json.loads(Path(args.report).read_text(encoding="utf-8")))
    print("valid" if not problems else "INVALID:\n  " + "\n  ".join(problems))
    return 0 if not problems else 1


def cmd_replay(args: argparse.Namespace) -> int:
    from .inspect02 import replay_report02

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    problems = replay_report02(report, load_observation(args.image, args.mask), _library(args.library))
    print("replay reproduced the report" if not problems else "MISMATCH:\n  " + "\n  ".join(problems))
    return 0 if not problems else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m examples.hand_anatomy_02.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("benchmark", help="score calibration split; --final scores the test split once")
    s.add_argument("--output", default=OUT + "/benchmark")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--final", action="store_true")
    s.add_argument("--reuse-records", action="store_true", help="re-evaluate saved per-case records")
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("pair-sheet", help="valid-vs-defect overlays for sample test pairs")
    s.add_argument("--output", default=OUT + "/pair_sheet.png")
    s.add_argument("--count", type=int, default=6)
    s.add_argument("--library")
    s.add_argument("--contract")
    s.set_defaults(fn=cmd_pair_sheet)

    def inputs(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--image")
        sp.add_argument("--mask")
        sp.add_argument("--library")

    s = sub.add_parser("inspect", help="inspect an image/mask with the residual VPM pipeline")
    inputs(s)
    s.add_argument("--contract", help="calibrated_contract02.json from a benchmark run")
    s.add_argument("--output", required=True)
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("validate-report")
    s.add_argument("--report", required=True)
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("replay")
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
