# zeromodel/pipeline/provenance/vpf_encode.py
from __future__ import annotations
import numpy as np
from typing import Any, Dict, Tuple
from zeromodel.pipeline.base import PipelineStage

# your module provides create_vpf (dict in/dict out)
from zeromodel.provenance.core import create_vpf

class VPFEncode(PipelineStage):
    """
    Build a VPF dict from the current pipeline context and attach it to context['vpf'].
    Expects any row/col orders to be present in context (e.g., from TopLeft stage).
    """
    name = "vpf_encode"
    category = "provenance"

    def __init__(self, **params):
        super().__init__(**params)
        self.task = params.get("task", "zeromodel-pipeline")
        self.model_id = params.get("model_id", "zero-1.0")

    def validate_params(self):
        pass

    def process(self, vpm: np.ndarray, context: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctx = self.get_context(context)

        # pull what we can from ctx (soft fallbacks)
        row_order = ctx.get("row_order") or []
        col_order = ctx.get("col_order") or []
        total_docs = int(vpm.shape[0]) if vpm.ndim >= 2 else 0
        total_metrics = int(vpm.shape[-1]) if vpm.ndim >= 2 else 0

        # minimal VPF
        vpf = create_vpf(
            pipeline={"graph_hash": f"sha3:{self.task}", "step": ctx.get("current_step", "organize"), "step_schema_hash": "sha3:zeromodel-v1"},
            model={"id": self.model_id, "assets": {}},
            determinism={"seed_global": 0, "seed_sampler": 0, "rng_backends": ["numpy"]},
            params={"task": self.task, "doc_order": row_order, "metric_order": col_order, "size": [int(total_metrics), int(total_docs)]},
            inputs={"task": self.task},
            metrics={"documents": total_docs, "metrics": total_metrics, "top_doc_global": row_order[0] if row_order else 0},
            lineage={"parents": [], "content_hash": "", "vpf_hash": ""},
        )

        ctx["vpf"] = vpf
        meta = {"vpf_keys": sorted(vpf.keys()), "doc_order_len": len(row_order), "metric_order_len": len(col_order)}
        return vpm, meta
