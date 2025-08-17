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

    def _load_stage(self, stage_path: str, params: Dict[str, Any]):
        if '.' in stage_path:
            pkg, clsname = stage_path.rsplit('.', 1)
        else:
            pkg, clsname = stage_path, 'Stage'
        module_path = f"zeromodel.pipeline.{pkg.replace('/', '.')}"
        module = importlib.import_module(module_path)
        cls = getattr(module, clsname)
        if not issubclass(cls, PipelineStage):
            raise TypeError(f"{cls} is not a PipelineStage")
        return cls(**params)

    def _init_context(self, context: Dict[str, Any] | None) -> Dict[str, Any]:
        ctx = {} if context is None else dict(context)
        ctx.setdefault('provenance', [])
        ctx.setdefault('pipeline_start_time', np.datetime64('now'))
        return ctx

    def run(self, vpm: np.ndarray, context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctx = self._init_context(context)
        cur = vpm

        for i, spec in enumerate(self.stages):
            stage_path = spec["stage"]
            params = spec.get("params", {})
            ctx['provenance'].append({
                'stage': stage_path,
                'params': params,
                'timestamp': np.datetime64('now'),
                'stage_index': i
            })
            try:
                stage = self._load_stage(stage_path, params)
                t0 = time.time()
                stage.validate_params()
                cur, meta = stage.process(cur, ctx)
                dt = time.time() - t0
                ctx[f'stage_{i}_metadata'] = {**(meta or {}), 'execution_time': dt, 'stage': stage_path}
                logger.debug(f"{stage_path} ok in {dt:.3f}s")
            except Exception as e:
                logger.exception(f"Stage {stage_path} failed")
                ctx[f'stage_{i}_error'] = {'stage': stage_path, 'error': str(e), 'timestamp': np.datetime64('now')}
                # passthrough cur

        ctx['final_stats'] = {
            'vpm_shape': tuple(cur.shape),
            'vpm_min': float(np.min(cur)),
            'vpm_max': float(np.max(cur)),
            'vpm_mean': float(np.mean(cur)),
            'pipeline_stages': len(self.stages),
            'total_execution_time': sum(ctx.get(f'stage_{i}_metadata', {}).get('execution_time', 0.0)
                                        for i in range(len(self.stages)))
        }
        return cur, ctx
