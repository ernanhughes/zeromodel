# zeromodel/pipeline/executor.py
import importlib
import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)

class PipelineExecutor:
    def __init__(self, stages: List[Dict[str, Any]]):
        self.stages = stages
        logger.info(f"PipelineExecutor initialized with {len(stages)} stages")

    def _load_stage(self, stage_path: str, params: Dict[str, Any]) -> PipelineStage:
        # Supports "pkg/subpkg.ClassName" or "pkg/subpkg" (uses first PipelineStage subclass)
        if "." in stage_path:
            pkg, clsname = stage_path.rsplit(".", 1)
        else:
            pkg, clsname = stage_path, None

        module_path = f"zeromodel.pipeline.{pkg.replace('/', '.')}"
        module = importlib.import_module(module_path)

        cls = getattr(module, clsname) if clsname else None
        if cls is None:
            # fallback: pick first subclass of PipelineStage in the module
            for attr in module.__dict__.values():
                if isinstance(attr, type) and issubclass(attr, PipelineStage) and attr is not PipelineStage:
                    cls = attr
                    break
        if cls is None:
            raise ImportError(f"No PipelineStage subclass found in {module_path}")

        inst = cls(**params)
        return inst

    def _init_context(self, context: Dict[str, Any] | None) -> Dict[str, Any]:
        ctx = {} if context is None else dict(context)
        ctx.setdefault("provenance", [])
        ctx.setdefault("pipeline_start_time", np.datetime64("now"))
        ctx.setdefault("stats", {})
        return ctx

    def _record(self, ctx: Dict[str, Any], **event):
        ctx["provenance"].append({"timestamp": np.datetime64("now"), **event})

    def run(self, vpm: np.ndarray, context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctx = self._init_context(context)
        cur = vpm

        for i, spec in enumerate(self.stages):
            stage_path = spec["stage"]
            params = spec.get("params", {})

            # stage start event
            self._record(ctx,
                kind="stage_start",
                stage=stage_path,
                index=i,
                params=params,
                input_shape=tuple(cur.shape),
            )

            t0 = time.time()
            try:
                stage = self._load_stage(stage_path, params)
                stage.validate_params()

                out, meta = stage.process(cur, ctx)
                dt = time.time() - t0

                # stage end (success)
                self._record(ctx,
                    kind="stage_end",
                    stage=stage_path,
                    index=i,
                    ok=True,
                    elapsed_sec=dt,
                    output_shape=tuple(out.shape),
                    metadata=meta or {},
                )

                # per-stage convenience block
                ctx[f"stage_{i}"] = {
                    "stage": stage_path,
                    "params": params,
                    "elapsed_sec": dt,
                    "input_shape": tuple(cur.shape),
                    "output_shape": tuple(out.shape),
                    "metadata": meta or {},
                }
                cur = out

            except Exception as e:
                dt = time.time() - t0
                logger.exception(f"Stage {stage_path} failed")
                # stage end (failure)
                self._record(ctx,
                    kind="stage_end",
                    stage=stage_path,
                    index=i,
                    ok=False,
                    elapsed_sec=dt,
                    error=str(e),
                )
                ctx[f"stage_{i}_error"] = {
                    "stage": stage_path,
                    "error": str(e),
                    "elapsed_sec": dt,
                    "timestamp": np.datetime64("now"),
                }
                # passthrough: keep cur unchanged and continue

        # final stats
        ctx["final_stats"] = {
            "vpm_shape": tuple(cur.shape),
            "vpm_min": float(np.min(cur)),
            "vpm_max": float(np.max(cur)),
            "vpm_mean": float(np.mean(cur)),
            "pipeline_stages": len(self.stages),
            "total_execution_time": float(sum(
                ctx.get(f"stage_{i}", {}).get("elapsed_sec", 0.0)
                for i in range(len(self.stages))
            )),
        }
        return cur, ctx
