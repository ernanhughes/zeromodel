"""Benchmark designed to be able to disprove the hypothesis.

* Fixed calibration / test splits (disjoint ids and masks, verified).
* Baseline A: best template by IoU alone (same library, same alignment).
* Method B: structured evidence compiled through the VPM + decision contract.
* Thresholds for BOTH are chosen on the calibration split only.
* Failures (false negatives/positives, abstentions) are preserved.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import _bootstrap  # noqa: F401
from zeromodel.core.render import png_bytes

from ._common import digest
from .assessment import SUPPORTED, SUSPECT, DecisionContract, decide
from .evidence import build_report, inspect_observation, replay_report, validate_report
from .observations import observation_from_arrays
from .pose_library import TemplateLibrary, build_library
from .synthetic import Case, build_cases
from .vpm_compiler import evidence_rows_from_vpm, observation_level_from_vpm

BENCHMARK_VERSION = "hand-benchmark/1"
FPR_CAP = 0.10

_LIB: Optional[TemplateLibrary] = None


def _library() -> TemplateLibrary:
    global _LIB
    if _LIB is None:
        _LIB = build_library()
    return _LIB


def evaluate_case(case: Case) -> Dict[str, Any]:
    """Run the full pipeline once; keep RAW evidence so decisions can be re-derived."""
    lib = _library()
    t0 = time.perf_counter()
    obs = observation_from_arrays(case.mask, method={"mask_source": "synthetic-case", "case_id": case.case_id})
    res = inspect_observation(obs, lib)
    elapsed = time.perf_counter() - t0
    vpm = res.compiled.vpm
    rows = evidence_rows_from_vpm(vpm)
    # decision directly from candidate evidence (no VPM) for the equivalence check
    direct_rows = sorted(
        (
            {"template_id": c.template_id, "name": c.template_name,
             "protrusions_template": c.diagnostics["protrusions_template"], "metrics": c.metrics}
            for c in res.candidates
        ),
        key=lambda r: tuple(vpm.source.row_ids).index(r["template_id"]),
    )
    by_view = {r["template_id"]: r for r in direct_rows}
    direct_in_view_order = [by_view[r["template_id"]] for r in rows]
    t1 = time.perf_counter()
    from_vpm = decide(rows, observation_level_from_vpm(vpm), DecisionContract())
    decide_us = (time.perf_counter() - t1) * 1e6
    from_direct = decide(direct_in_view_order, dict(observation_level_from_vpm(vpm)), DecisionContract())
    cand = {c.template_id: c for c in res.candidates}
    top = rows[0]
    return {
        "case_id": case.case_id, "split": case.split, "group": case.group, "kind": case.kind,
        "label": case.label, "ambiguous": case.ambiguous, "mask_id": case.mask_id,
        "rows": rows, "obs_level": dict(observation_level_from_vpm(vpm)),
        "best_template": top["name"],
        "best_iou": top["metrics"]["silhouette_agreement"],
        "max_iou": max(r["metrics"]["silhouette_agreement"] for r in rows),
        "largest_unexplained": cand[top["template_id"]].diagnostics.get("largest_unexplained"),
        "runtime_s": elapsed, "decide_us": decide_us,
        "vpm_json_bytes": len(json.dumps(vpm.to_dict())), "vpm_png_bytes": len(png_bytes(vpm)),
        "vpm_matches_direct": from_vpm["status"] == from_direct["status"]
        and from_vpm["contradictions"] == from_direct["contradictions"],
        "vpm_id": vpm.artifact_id,
    }


# ------------------------------------------------------------------ scoring
def wilson(k: int, n: int, z: float = 1.96) -> Optional[List[float]]:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _ratio(a: int, b: int) -> Optional[float]:
    return None if b == 0 else round(a / b, 4)


def score(records: Sequence[Dict[str, Any]], preds: Sequence[str]) -> Dict[str, Any]:
    """preds in {"pos", "neg", "abstain"}; positive class = malformed."""
    tp = fp = tn = fn = ab_pos = ab_neg = 0
    for r, p in zip(records, preds):
        mal = r["label"] == "malformed"
        if p == "abstain":
            ab_pos += mal
            ab_neg += (not mal)
        elif p == "pos":
            tp += mal
            fp += (not mal)
        else:
            tn += (not mal)
            fn += mal
    n = len(records)
    n_mal, n_val = tp + fn + ab_pos, tn + fp + ab_neg
    decided = tp + fp + tn + fn
    prec, rec = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = None if (prec is None or rec is None or prec + rec == 0) else round(2 * prec * rec / (prec + rec), 4)
    return {
        "n": n, "n_malformed": n_mal, "n_valid": n_val,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "abstained_malformed": ab_pos, "abstained_valid": ab_neg,
        "precision": prec, "recall_decided": rec, "f1_decided": f1,
        "recall_overall": _ratio(tp, n_mal), "recall_overall_ci95": wilson(tp, n_mal),
        "fpr_decided": _ratio(fp, fp + tn), "fpr_overall": _ratio(fp, n_val),
        "fpr_decided_ci95": wilson(fp, fp + tn),
        "abstention_rate": _ratio(ab_pos + ab_neg, n), "coverage": _ratio(decided, n),
        "accuracy_on_accepted": _ratio(tp + tn, decided),
        "confusion": {"pred_pos": {"malformed": tp, "valid": fp}, "pred_neg": {"malformed": fn, "valid": tn},
                      "abstain": {"malformed": ab_pos, "valid": ab_neg}},
    }


def baseline_preds(records: Sequence[Dict[str, Any]], threshold: float) -> List[str]:
    return ["pos" if r["best_iou"] < threshold else "neg" for r in records]


def method_preds(records: Sequence[Dict[str, Any]], contract: DecisionContract) -> List[str]:
    out = []
    for r in records:
        s = decide(r["rows"], r["obs_level"], contract)["status"]
        out.append({SUSPECT: "pos", SUPPORTED: "neg"}.get(s, "abstain"))
    return out


def _objective(records: Sequence[Dict[str, Any]], preds: Sequence[str]) -> Tuple[float, float]:
    """(feasible-F1, coverage) under the FPR cap; abstained malformed count as misses."""
    s = score(records, preds)
    fpr = s["fpr_decided"] or 0.0
    tp, fp = s["tp"], s["fp"]
    fn = s["n_malformed"] - tp  # abstained malformed are misses
    f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return (f1 if fpr <= FPR_CAP else -1.0 + f1 * 0.01, s["coverage"] or 0.0)


def calibrate_baseline(cal: Sequence[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    best = None
    for t in [round(x, 3) for x in np.arange(0.40, 0.99, 0.01)]:
        key = (_objective(cal, baseline_preds(cal, t)), -t)
        if best is None or key > best[0]:
            best = (key, t)
    assert best is not None
    return best[1], {"objective_f1": best[0][0][0], "fpr_cap": FPR_CAP}


GRID = {
    "fit_floor": (0.45, 0.55, 0.65),
    "fit_ok": (0.70, 0.78, 0.85),
    "protrusion_floor": (0.2, 0.35, 0.5, 0.65, 0.8),
    "observed_floor": (0.88, 0.92, 0.95, 0.98),
    "missing_floor": (0.75, 0.85, 0.92),
    "use_structure": (False, True),
}


def calibrate_method(cal: Sequence[Dict[str, Any]], split_id: str) -> Tuple[DecisionContract, Dict[str, Any]]:
    best = None
    names = list(GRID)
    for combo in itertools.product(*(GRID[n] for n in names)):
        c = DecisionContract(**dict(zip(names, combo)))
        key = (_objective(cal, method_preds(cal, c)), tuple(-v if isinstance(v, float) else -int(v) for v in combo))
        if best is None or key > best[0]:
            best = (key, c)
    assert best is not None
    obj = best[0][0]
    contract = best[1].with_values(
        calibration={"split_id": split_id, "objective_f1": round(obj[0], 4), "coverage": round(obj[1], 4),
                     "fpr_cap": FPR_CAP, "grid": {k: list(v) for k, v in GRID.items()}}
    )
    return contract, {"objective_f1": obj[0], "coverage": obj[1]}


# ------------------------------------------------------------------ driver
def run_cases(cases: Sequence[Case], workers: int) -> List[Dict[str, Any]]:
    if workers <= 1:
        return [evaluate_case(c) for c in cases]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(evaluate_case, cases, chunksize=4))


def leakage_checks(splits: Dict[str, List[Case]], library: TemplateLibrary) -> Dict[str, Any]:
    cal_ids = {c.case_id for c in splits["calibration"]}
    test_ids = {c.case_id for c in splits["test"]}
    cal_masks = {c.mask_id for c in splits["calibration"]}
    test_masks = {c.mask_id for c in splits["test"]}
    lib_masks = {t.mask_id for t in library.templates}
    out = {
        "case_id_overlap": len(cal_ids & test_ids),
        "mask_overlap_calibration_test": len(cal_masks & test_masks),
        "mask_overlap_test_library": len(test_masks & lib_masks),
        "mask_overlap_calibration_library": len(cal_masks & lib_masks),
        "library_built_from_cases": False,
    }
    out["ok"] = all(out[k] == 0 for k in out if k.endswith("overlap") or "overlap" in k)
    return out


def _subset(records, pred):
    return [i for i, r in enumerate(records) if pred(r)]


def summarize(records: Sequence[Dict[str, Any]], preds: Sequence[str]) -> Dict[str, Any]:
    def sub(f):
        idx = _subset(records, f)
        return score([records[i] for i in idx], [preds[i] for i in idx])

    out = {
        "all": score(records, preds),
        "independent_clear": sub(lambda r: r["group"] == "independent" and not r["ambiguous"]),
        "independent_ambiguous_only": sub(lambda r: r["group"] == "independent" and r["ambiguous"]),
        "same_renderer_weak": sub(lambda r: r["group"] == "same_renderer"),
    }
    per_kind: Dict[str, Any] = {}
    for kind in sorted({r["kind"] for r in records}):
        for group in ("independent", "same_renderer"):
            idx = _subset(records, lambda r, k=kind, g=group: r["kind"] == k and r["group"] == g)
            if not idx:
                continue
            c = {"pos": 0, "neg": 0, "abstain": 0}
            for i in idx:
                c[preds[i]] += 1
            per_kind["%s/%s" % (group, kind)] = {
                "label": records[idx[0]]["label"], "n": len(idx), "flagged": c["pos"],
                "passed": c["neg"], "abstained": c["abstain"],
            }
    out["per_kind"] = per_kind
    return out


def run_benchmark(output: str | Path, workers: int = 4, replay_samples: int = 3) -> Dict[str, Any]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    lib = _library()
    splits = build_cases()
    leak = leakage_checks(splits, lib)
    cases = splits["calibration"] + splits["test"]
    records = run_cases(cases, workers)
    cal = [r for r in records if r["split"] == "calibration"]
    test = [r for r in records if r["split"] == "test"]
    split_id = digest({"calibration": sorted(r["case_id"] for r in cal)})
    base_t, base_cal = calibrate_baseline(cal)
    contract, meth_cal = calibrate_method(cal, split_id)
    methods = {
        "A_silhouette_iou_baseline": (baseline_preds(test, base_t), {"iou_threshold": base_t, **base_cal}),
        "B_structured_evidence_calibrated": (method_preds(test, contract), {"contract": contract.to_dict()}),
        "B0_structured_evidence_uncalibrated_defaults": (method_preds(test, DecisionContract()), {"contract": DecisionContract().to_dict()}),
        "B1_unexplained_protrusion_only": (
            method_preds(test, contract.with_values(use_observed_area=False, use_structure=False, calibration=None)),
            {"note": "calibrated floors, protrusion rule only (ablation; floors not re-fit)"}),
        "B2_without_structure_heuristic": (
            method_preds(test, contract.with_values(use_structure=False, calibration=None)),
            {"note": "calibrated floors, structure heuristic disabled (ablation)"}),
    }
    results = {name: {"parameters": params, "test": summarize(test, preds)} for name, (preds, params) in methods.items()}
    results["B_structured_evidence_calibrated"]["calibration_split_scores"] = summarize(
        cal, method_preds(cal, contract))["all"]
    results["A_silhouette_iou_baseline"]["calibration_split_scores"] = summarize(
        cal, baseline_preds(cal, base_t))["all"]
    # traceability: replay a few real reports
    replay = {"checked": 0, "validated": 0, "replayed": 0, "problems": []}
    case_by_id = {c.case_id: c for c in cases}
    for r in test[:: max(1, len(test) // max(1, replay_samples))][:replay_samples]:
        c = case_by_id[r["case_id"]]
        obs = observation_from_arrays(c.mask, method={"mask_source": "synthetic-case", "case_id": c.case_id})
        res = inspect_observation(obs, lib, contract)
        rep = json.loads(json.dumps(build_report(res)))
        replay["checked"] += 1
        pv = validate_report(rep)
        replay["validated"] += not pv
        pr = replay_report(rep, obs, lib)
        replay["replayed"] += not pr
        replay["problems"] += pv + pr
    # failure preservation
    fail_dir = out / "failures"
    fail_dir.mkdir(exist_ok=True)
    failures: List[Dict[str, Any]] = []
    for r, p in zip(test, methods["B_structured_evidence_calibrated"][0]):
        mal = r["label"] == "malformed"
        kind = None
        if mal and p == "neg":
            kind = "false_negative"
        elif mal and p == "abstain":
            kind = "abstained_malformed"
        elif not mal and p == "pos":
            kind = "false_positive"
        elif not mal and p == "abstain":
            kind = "abstained_valid"
        if kind:
            failures.append({"type": kind, "case_id": r["case_id"], "group": r["group"], "kind": r["kind"],
                             "ambiguous": r["ambiguous"], "best_iou": round(r["best_iou"], 4),
                             "best_template": r["best_template"],
                             "reasons": decide(r["rows"], r["obs_level"], contract)["reasons"]})
            if len(failures) <= 80:
                Image.fromarray((case_by_id[r["case_id"]].mask * 255).astype(np.uint8), "L").save(
                    fail_dir / ("%s_%s_%s.png" % (kind, r["kind"], r["case_id"][7:15])))
    runtimes = [r["runtime_s"] for r in records]
    meta = {
        "schema_version": BENCHMARK_VERSION,
        "library": {"library_id": lib.library_id, "size": len(lib)},
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "calibration_split_id": split_id,
        "leakage_checks": leak,
        "fpr_cap_for_calibration": FPR_CAP,
        "vpm_equivalence": {
            "cases": len(records),
            "decision_from_vpm_equals_decision_from_evidence": sum(r["vpm_matches_direct"] for r in records),
        },
        "runtime": {"mean_case_s": round(float(np.mean(runtimes)), 3), "max_case_s": round(float(np.max(runtimes)), 3),
                    "mean_decide_us": round(float(np.mean([r["decide_us"] for r in records])), 1),
                    "total_wall_s": round(time.perf_counter() - t_start, 1), "workers": workers},
        "artifact_sizes": {"mean_vpm_json_bytes": int(np.mean([r["vpm_json_bytes"] for r in records])),
                           "mean_vpm_png_bytes": int(np.mean([r["vpm_png_bytes"] for r in records]))},
        "replay_traceability": replay,
        "failure_count_method_B": len(failures),
        "notes": [
            "Positive class = malformed. Abstentions are excluded from tp/fp/tn/fn and reported separately.",
            "Calibrated thresholds were selected on the calibration split only; test split untouched until scoring.",
            "Synthetic masks only; no real-image evidence.",
        ],
    }
    final = {"meta": meta, "results": results, "failures": failures}
    (out / "calibrated_contract.json").write_text(
        json.dumps({"contract": contract.to_dict(), "baseline_iou_threshold": base_t}, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    (out / "results.json").write_text(json.dumps(final, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    with (out / "per_case.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            slim = {k: v for k, v in r.items() if k != "rows"}
            slim["top_row_metrics"] = r["rows"][0]["metrics"]
            fh.write(json.dumps(slim, sort_keys=True) + "\n")
    (out / "results.md").write_text(render_markdown(final), encoding="utf-8")
    return final


def _fmt(v: Any) -> str:
    return "-" if v is None else ("%.3f" % v if isinstance(v, float) else str(v))


def render_markdown(final: Dict[str, Any]) -> str:
    lines = ["# ZM-HAND-01 benchmark results (synthetic masks only)", ""]
    m = final["meta"]
    lines += ["Library: %d templates. Split sizes: %s. Leakage checks ok: %s." % (
        m["library"]["size"], m["split_sizes"], m["leakage_checks"]["ok"]), ""]
    cols = ["tp", "fp", "tn", "fn", "abstained_malformed", "abstained_valid", "precision", "recall_overall",
            "fpr_decided", "coverage", "accuracy_on_accepted"]
    for subset in ("all", "independent_clear", "independent_ambiguous_only", "same_renderer_weak"):
        lines += ["## Test split: %s" % subset, "", "| method | " + " | ".join(cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for name, res in final["results"].items():
            s = res["test"][subset]
            lines.append("| %s | %s |" % (name, " | ".join(_fmt(s[c]) for c in cols)))
        lines.append("")
    lines += ["## Per-kind outcomes for B (calibrated), test split", "",
              "| group/kind | label | n | flagged | passed | abstained |", "|---|---|---|---|---|---|"]
    for k, v in final["results"]["B_structured_evidence_calibrated"]["test"]["per_kind"].items():
        lines.append("| %s | %s | %d | %d | %d | %d |" % (k, v["label"], v["n"], v["flagged"], v["passed"], v["abstained"]))
    lines += ["", "## Per-kind outcomes for A (IoU baseline), test split", "",
              "| group/kind | label | n | flagged | passed | abstained |", "|---|---|---|---|---|---|"]
    for k, v in final["results"]["A_silhouette_iou_baseline"]["test"]["per_kind"].items():
        lines.append("| %s | %s | %d | %d | %d | %d |" % (k, v["label"], v["n"], v["flagged"], v["passed"], v["abstained"]))
    return "\n".join(lines) + "\n"
