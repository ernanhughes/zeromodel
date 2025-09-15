#  zeromodel/pipeline/organizer/expose_orders.py
from __future__ import annotations
import numpy as np
from typing import Any, Dict, Tuple
from zeromodel.pipeline.base import PipelineStage

class ExposeOrders(PipelineStage):
    """
    No-op that just records current row/col order into context for later VPF encode.
    Provide them via params or prior stage's metadata in context.
    """
    name = "expose_orders"
    category = "organizer"

    def __init__(self, **params):
        super().__init__(**params)
        self.row_order = params.get("row_order")  # list[int] or None
        self.col_order = params.get("col_order")

    def process(self, vpm: np.ndarray, context: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctx = self._get_context(context)
        if self.row_order is not None:
            ctx["row_order"] = list(self.row_order)
        if self.col_order is not None:
            ctx["col_order"] = list(self.col_order)
        return vpm, {"reordering_applied": False, "row_order_len": len(ctx.get("row_order", [])), "col_order_len": len(ctx.get("col_order", []))}
