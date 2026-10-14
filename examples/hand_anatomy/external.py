"""Importer + evaluator for manually annotated external hand crops.

Layout (see fixtures/README.md): one folder per case containing ``image.png``,
``mask.png``, ``label.json`` and optionally ``landmarks.json``.  No result in
RESULTS.md uses this path; it exists so real generated-frame crops can be
scored with the same contract and baseline as the synthetic benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._common import UnsupportedInput
from .assessment import SUPPORTED, SUSPECT, DecisionContract
from .benchmark import baseline_preds, score
from .evidence import inspect_observation
from .observations import load_observation
from .pose_library import TemplateLibrary

LABELS = ("valid", "malformed")
MAX_CASES = 500


def import_cases(root: str | Path) -> List[Dict[str, Any]]:
    base = Path(root)
    if not base.is_dir():
        raise UnsupportedInput("external case root not found: %s" % base)
    cases = []
    for folder in sorted(p for p in base.iterdir() if p.is_dir())[:MAX_CASES]:
        label_path = folder / "label.json"
        if not label_path.is_file():
            raise UnsupportedInput("%s: missing label.json" % folder.name)
        meta = json.loads(label_path.read_text(encoding="utf-8"))
        if meta.get("label") not in LABELS:
            raise UnsupportedInput("%s: label must be one of %s" % (folder.name, LABELS))
        for required in ("image.png", "mask.png"):
            if not (folder / required).is_file():
                raise UnsupportedInput("%s: missing %s" % (folder.name, required))
        cases.append({
            "case": folder.name, "label": meta["label"], "defect": meta.get("defect"),
            "ambiguous": bool(meta.get("ambiguous", False)), "source": meta.get("source"),
            "image": str(folder / "image.png"), "mask": str(folder / "mask.png"),
            "landmarks": str(folder / "landmarks.json") if (folder / "landmarks.json").is_file() else None,
        })
    if not cases:
        raise UnsupportedInput("no cases found under %s" % base)
    return cases


def evaluate_external(
    root: str | Path, library: TemplateLibrary, contract: DecisionContract,
    baseline_threshold: float, output: Optional[str | Path] = None,
) -> Dict[str, Any]:
    cases = import_cases(root)
    records: List[Dict[str, Any]] = []
    preds: List[str] = []
    for c in cases:
        try:
            obs = load_observation(c["image"], c["mask"], c["landmarks"])
        except UnsupportedInput as exc:
            records.append({**c, "label": c["label"], "status": "unsupported_input", "reason": str(exc), "best_iou": 0.0})
            preds.append("abstain")
            continue
        res = inspect_observation(obs, library, contract)
        status = res.decision["status"]
        records.append({**c, "status": status, "reasons": res.decision["reasons"],
                        "best_iou": res.decision["best_fit"] if "best_fit" in res.decision else 0.0,
                        "vpm_artifact_id": res.compiled.vpm.artifact_id})
        preds.append({SUSPECT: "pos", SUPPORTED: "neg"}.get(status, "abstain"))
    out = {
        "n_cases": len(cases),
        "structured_evidence": score(records, preds),
        "iou_baseline": score(records, baseline_preds(records, baseline_threshold)),
        "cases": [{k: v for k, v in r.items() if k not in ("image", "mask", "landmarks")} for r in records],
    }
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "external_results.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out
