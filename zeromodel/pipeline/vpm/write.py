#  zeromodel/pipeline/vpm/write.py
from __future__ import annotations
import os
import time
import zlib
from typing import Any, Dict, Optional, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage
from zeromodel.vpm.image import VPMImageWriter
from zeromodel.vpm.metadata import AggId, VPMMetadata


class VPMWrite(PipelineStage):
    name = "vpm_write"
    category = "io"

    def __init__(self, **params):
        super().__init__(**params)
        self.output_path: Optional[str] = params.get("output_path")

    def validate_params(self):
        if not self.output_path:
            raise ValueError("VPMWriteStage requires output_path")

    def process(self, vpm: np.ndarray, context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctx = self.get_context(context)
        metric_names = ctx.get("metric_names", [f"m{i}" for i in range(vpm.shape[1])])
        # choose source: sorted_matrix if present else vpm
        source = ctx.get("sorted_matrix", vpm)
        mx_d = source.T
        logical_docs = int(mx_d.shape[1])

        # enforce min width 12
        MIN_W = 12
        if mx_d.shape[1] < MIN_W:
            pad = MIN_W - mx_d.shape[1]
            mx_d = np.pad(mx_d, ((0,0),(0,pad)), constant_values=0.0)

        sql_query = (ctx.get("task_config", {}) or {}).get("sql_query") or ""
        try:
            task_hash = zlib.crc32(sql_query.encode("utf-8")) & 0xFFFFFFFF
        except Exception:
            task_hash = 0

        tile_id = VPMMetadata.make_tile_id(f"{ctx.get('task','')!s}|{mx_d.shape}".encode("utf-8"))

        vmeta = VPMMetadata.for_tile(
            level=0,
            metric_count=int(mx_d.shape[0]),
            doc_count=logical_docs,
            doc_block_size=1,
            agg_id=int(AggId.RAW),
            metric_weights=None,
            metric_names=metric_names,
            task_hash=int(task_hash),
            tile_id=tile_id,
            parent_id=b"\x00"*16
        )

        compress = int(os.getenv("ZM_PNG_COMPRESS", "6"))
        disable_prov = os.getenv("ZM_DISABLE_PROVENANCE") == "1"
        metadata_bytes = None if disable_prov else vmeta.to_bytes()

        writer = VPMImageWriter(
            score_matrix=mx_d,
            metric_names=metric_names,
            metadata_bytes=metadata_bytes,
            store_minmax=True,
            compression=compress,
        )
        t0 = time.perf_counter()
        writer.write(self.output_path)
        io_dt = time.perf_counter() - t0

        ctx["vpm_image_path"] = self.output_path
        meta = {
            "output_path": self.output_path,
            "metrics": int(mx_d.shape[0]),
            "docs_logical": logical_docs,
            "io_sec": io_dt,
            "provenance_embedded": not disable_prov,
        }
        return vpm, meta
