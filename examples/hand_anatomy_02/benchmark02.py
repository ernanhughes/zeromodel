"""Counterfactual-pair benchmark at matched false-positive rate.

Protocol (frozen in PREREGISTRATION.md before the test split is run):

1. Score every case once; scores are derived from RAW VPM cells.
2. For each method, threshold = smallest value whose false-positive rate on the
   CALIBRATION valid pool (valid twins + valid pool) is <= FPR_CAP.
3. Apply thresholds unchanged to the test split.  Compare recall on the primary set
   (distinguishable extra-finger pairs) with an exact two-sided sign test on
   discordant pairs.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from examples.hand_anatomy.observations import observation_from_arrays
from examples.hand_anatomy.pose_library import TemplateLibrary, build_library

from .inspect02 import inspect02
from .pairs import PRIMARY_SUBTYPES, Case02, build_dataset
from .vpm02 import Contract02, rows_from_vpm02, scores_from_rows

FPR_CAP = 0.10
RULE = {
    "primary_set": "extra_finger pairs, subtype in outer/between/adjacent, not indistinguishable",
    "min_primary_pairs": 40,
    "min_recall_gain_vs_iou": 0.15,  # must hold against BOTH baselines
    "max_p_value_sign_test": 0.05,
    "max_test_fpr": FPR_CAP + 0.05,
}
METHODS = ("A_iou", "A_refined_iou", "B01_protrusion", "C_residual_unrefined", "C_residual", "C_branch")
BASELINES = ("A_iou", "A_refined_iou")

_LIB: Optional[TemplateLibrary] = None


def _library() -> TemplateLibrary:
    global _LIB
    if _LIB is None:
        _LIB = build_library()
    return _LIB


def evaluate_case02(case: Case02) -> Dict[str, Any]:
    t0 = time.perf_counter()
    obs = observation_from_arrays(case.mask, method={"mask_source": "synthetic-case", "case_id": case.case_id})
    res = inspect02(obs, _library(), Contract02())
    rows = rows_from_vpm02(res.vpm)
    return {
        "case_id": case.case_id, "split": case.split, "role": case.role, "label": case.label,
        "pair_id": case.pair_id, "category": case.category, "subtype": case.subtype,
        "indistinguishable": case.indistinguishable, "mask_id": case.mask_id,
        "rows": [{"template_id": r["template_id"], "name": r["name"], "metrics": r["metrics"]} for r in rows],
        "obs_level": dict(res.vpm.source.metadata["observation_level"]),
        "top_residual": res.residual_diag[res.explainer_template_id]["top_residual"],
        "runtime_s": time.perf_counter() - t0, "vpm_id": res.vpm.artifact_id,
    }


def run_cases(cases: Sequence[Case02], workers: int) -> List[Dict[str, Any]]:
    if workers <= 1:
        return [evaluate_case02(c) for c in cases]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(evaluate_case02, cases, chunksize=4))


# ------------------------------------------------------------------ scoring
def scores(rows: Sequence[Dict[str, Any]], fit_floor: float = 0.45, margin: Optional[float] = None) -> Dict[str, float]:
    """Anomaly scores from RAW VPM rows (``margin`` is unused; kept for signature stability)."""
    return scores_from_rows(rows, fit_floor)


def threshold_at_fpr(valid_scores: Sequence[float], cap: float) -> float:
    """Smallest t such that #(score > t) <= floor(cap * n)."""
    s = sorted(valid_scores, reverse=True)
    allowed = int(math.floor(cap * len(s)))
    return s[allowed] if allowed < len(s) else -1.0


def sign_test_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k: int, n: int, z: float = 1.96) -> Optional[List[float]]:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def evaluate(cal: Sequence[Dict[str, Any]], test: Sequence[Dict[str, Any]], fit_floor: float = 0.45,
             margin: Optional[float] = None, methods: Sequence[str] = METHODS) -> Dict[str, Any]:
    cal_valid = [r for r in cal if r["label"] == "valid"]
    out: Dict[str, Any] = {"parameters": {"fit_floor": fit_floor, "margin": margin, "fpr_cap": FPR_CAP}, "methods": {}}
    sc_cal = {r["case_id"]: scores(r["rows"], fit_floor, margin) for r in cal}
    sc_test = {r["case_id"]: scores(r["rows"], fit_floor, margin) for r in test}
    t_valid = [r for r in test if r["label"] == "valid"]
    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in test:
        if r["pair_id"]:
            by_pair.setdefault(r["pair_id"], {})[r["role"]] = r
    pairs = list(by_pair.values())
    primary = [p for p in pairs if p["defect"]["category"] == "extra_finger"
               and p["defect"]["subtype"] in PRIMARY_SUBTYPES and not p["defect"]["indistinguishable"]]
    for m in methods:
        thr = threshold_at_fpr([sc_cal[r["case_id"]][m] for r in cal_valid], FPR_CAP)
        det = lambda r: sc_test[r["case_id"]][m] > thr  # noqa: E731
        fp = sum(det(r) for r in t_valid)
        res: Dict[str, Any] = {
            "threshold": thr,
            "test_fpr": round(fp / len(t_valid), 4), "test_fpr_ci95": wilson(fp, len(t_valid)),
            "test_valid_n": len(t_valid), "by_group": {},
        }
        groups: Dict[str, List[Dict[str, Dict[str, Any]]]] = {}
        for p in pairs:
            d = p["defect"]
            key = "%s/%s%s" % (d["category"], d["subtype"], "[indistinguishable]" if d["indistinguishable"] else "")
            groups.setdefault(key, []).append(p)
        for key, ps in sorted(groups.items()):
            hits = sum(det(p["defect"]) for p in ps)
            sep = np.mean([(sc_test[p["defect"]["case_id"]][m] > sc_test[p["valid_twin"]["case_id"]][m])
                           + 0.5 * (sc_test[p["defect"]["case_id"]][m] == sc_test[p["valid_twin"]["case_id"]][m]) for p in ps])
            res["by_group"][key] = {"pairs": len(ps), "detected": hits, "recall": round(hits / len(ps), 4),
                                    "recall_ci95": wilson(hits, len(ps)), "paired_separation": round(float(sep), 4)}
        k = sum(det(p["defect"]) for p in primary)
        res["primary"] = {"pairs": len(primary), "detected": k, "recall": round(k / len(primary), 4) if primary else None,
                          "recall_ci95": wilson(k, len(primary))}
        out["methods"][m] = res
        res["_flags"] = [det(p["defect"]) for p in primary]
    for m in methods:
        if m in BASELINES:
            continue
        f = out["methods"][m]["_flags"]
        out["methods"][m]["vs_baselines_primary"] = {}
        ok = len(primary) >= RULE["min_primary_pairs"] and out["methods"][m]["test_fpr"] <= RULE["max_test_fpr"]
        for base in BASELINES:
            if base not in out["methods"]:
                continue
            a_flags = out["methods"][base]["_flags"]
            b_only = sum(1 for x, y in zip(f, a_flags) if x and not y)
            c_only = sum(1 for x, y in zip(f, a_flags) if y and not x)
            gain = (out["methods"][m]["primary"]["recall"] or 0.0) - (out["methods"][base]["primary"]["recall"] or 0.0)
            pval = sign_test_p(b_only, c_only)
            out["methods"][m]["vs_baselines_primary"][base] = {
                "recall_gain": round(gain, 4), "discordant_method_only": b_only,
                "discordant_baseline_only": c_only, "sign_test_p": round(pval, 6)}
            ok = ok and gain >= RULE["min_recall_gain_vs_iou"] and pval < RULE["max_p_value_sign_test"]
        out["methods"][m]["meets_preregistered_rule"] = bool(ok)
    for m in out["methods"].values():
        m.pop("_flags", None)
    out["unpaired_auc"] = {}
    for m in methods:
        pos = [sc_test[r["case_id"]][m] for r in test if r["label"] == "malformed" and not r["indistinguishable"]]
        neg = [sc_test[r["case_id"]][m] for r in t_valid]
        out["unpaired_auc"][m] = round(float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])), 4)
    return out


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(out: Path, cal: Sequence[Dict[str, Any]], test: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate saved per-case records under the frozen rule and write results + calibrated contract."""
    result: Dict[str, Any] = {"final": True, "rule": RULE, "preregistration_sha256": None}
    prereg = Path(__file__).with_name("PREREGISTRATION.md")
    if prereg.exists():
        result["preregistration_sha256"] = sha256_file(prereg)
    ev = evaluate(cal, test)
    result["evaluation"] = ev
    ok = ev["methods"]["C_residual"]["meets_preregistered_rule"]
    result["verdict_C_residual"] = "ACCEPT_FOR_EXTRA_FINGER_DETECTION" if ok else "REJECT"
    cal_valid = [r for r in cal if r["label"] == "valid"]
    thr = threshold_at_fpr([scores(r["rows"])["C_residual"] for r in cal_valid], FPR_CAP)
    contract = Contract02(residual_threshold=thr, calibration={
        "split": "cal02", "fpr_cap": FPR_CAP, "valid_cases": len(cal_valid),
        "note": "threshold set on calibration valid pool; contract is NOT validated for use"})
    (out / "calibrated_contract02.json").write_text(
        json.dumps({"contract": contract.to_dict()}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "results.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: Dict[str, Any]) -> str:
    ev = result["evaluation"]
    lines = ["# ZM-HAND-02 results (synthetic counterfactual pairs)", "",
             "Verdict for C_residual under the preregistered rule: **%s**" % result["verdict_C_residual"], "",
             "## Primary set (distinguishable extra-finger pairs) at calibration-matched FPR", "",
             "| method | threshold | test FPR | detected / pairs | recall | gain vs A_iou (p) | gain vs A_refined_iou (p) |",
             "|---|---|---|---|---|---|---|"]
    for m, v in ev["methods"].items():
        vb = v.get("vs_baselines_primary", {})
        def cell(b):
            return "%+.2f (p=%.3f)" % (vb[b]["recall_gain"], vb[b]["sign_test_p"]) if b in vb else "-"
        lines.append("| %s | %.2f | %.3f | %d / %d | %.2f | %s | %s |" % (
            m, v["threshold"], v["test_fpr"], v["primary"]["detected"], v["primary"]["pairs"],
            v["primary"]["recall"], cell("A_iou"), cell("A_refined_iou")))
    lines += ["", "Unpaired AUC (distinguishable defects vs valid pool): %s" % ev["unpaired_auc"], "",
              "## Per group: detected / pairs (paired separation = P[defect scores above its own valid twin])", ""]
    methods = ["A_iou", "A_refined_iou", "C_residual"]
    lines += ["| group | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for key in ev["methods"]["A_iou"]["by_group"]:
        cells = []
        for m in methods:
            g = ev["methods"][m]["by_group"][key]
            cells.append("%d/%d (sep %.2f)" % (g["detected"], g["pairs"], g["paired_separation"]))
        lines.append("| %s | %s |" % (key, " | ".join(cells)))
    return "\n".join(lines) + "\n"


def run_benchmark02(output: str | Path, workers: int = 4, final: bool = False,
                    reuse_records: bool = False) -> Dict[str, Any]:
    """Score cal02 (always) and test02 (only if ``final``); ``reuse_records`` re-evaluates saved records."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if reuse_records:
        cal = json.loads((out / "cal_records.json").read_text(encoding="utf-8"))
        test = json.loads((out / "test_records.json").read_text(encoding="utf-8"))
        return finalize(out, cal, test)
    ds = build_dataset()
    cal = run_cases(ds["cal02"]["cases"], workers)
    (out / "cal_records.json").write_text(json.dumps(cal), encoding="utf-8")
    if not final:
        return {"final": False, "cal_cases": len(cal), "wall_s": round(time.perf_counter() - t0, 1)}
    test = run_cases(ds["test02"]["cases"], workers)
    (out / "test_records.json").write_text(json.dumps(test), encoding="utf-8")
    result = finalize(out, cal, test)
    result["wall_s"] = round(time.perf_counter() - t0, 1)
    return result
