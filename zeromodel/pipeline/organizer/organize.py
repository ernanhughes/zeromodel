# zeromodel/pipeline/stage/organize.py
from typing import Any, Dict, Optional, Tuple

import numpy as np

from zeromodel.config import get_config
from zeromodel.organization import (DuckDBAdapter, MemoryOrganizationStrategy,
                                    SqlOrganizationStrategy)
from zeromodel.pipeline.base import PipelineStage


class Organize(PipelineStage):
    name = "organize"
    category = "organizer"

    def __init__(self, **params):
        super().__init__(**params)
        self.sql_query: Optional[str] = params.get("sql_query")

    def validate_params(self): pass

    def process(self, vpm: np.ndarray, context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctx = self.get_context(context)
        metric_names = ctx.get("metric_names", [f"m{i}" for i in range(vpm.shape[1])])
        use_duckdb = bool(get_config("core").get("use_duckdb", False))
        q = (self.sql_query or "").strip()

        if not q:
            metric_order = np.arange(vpm.shape[1], dtype=int)
            doc_order = np.arange(vpm.shape[0], dtype=int)
            sorted_matrix = vpm
            analysis = {"backend": "none", "reason": "no sql_query provided"}
            task = "noop_task"
        elif use_duckdb and q.lower().startswith("select "):
            duck = ctx.get("duckdb") or DuckDBAdapter(metric_names)
            ctx["duckdb"] = duck
            strat = SqlOrganizationStrategy(duck)
            strat.set_task(q)
            _, metric_order, doc_order, analysis = strat.organize(vpm, metric_names)
            sorted_matrix = vpm[doc_order][:, metric_order]
            task = strat.name + "_task"
        else:
            strat = MemoryOrganizationStrategy()
            strat.set_task(q)  # e.g. "metric DESC"
            _, metric_order, doc_order, analysis = strat.organize(vpm, metric_names)
            sorted_matrix = vpm[doc_order][:, metric_order]
            task = "memory_task"

        ctx["metric_order"] = metric_order
        ctx["doc_order"] = doc_order
        ctx["sorted_matrix"] = sorted_matrix
        ctx["task"] = task
        ctx["task_config"] = {"sql_query": self.sql_query, "analysis": analysis}

        meta = {
            "metric_order_len": int(metric_order.size),
            "doc_order_len": int(doc_order.size),
            "task": task,
            "shape": tuple(sorted_matrix.shape),
        }
        return sorted_matrix, meta
