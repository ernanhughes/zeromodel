# tests/test_core_pipeline_prepare.py
import numpy as np
from zeromodel.pipeline.executor import PipelineExecutor

import logging

logger = logging.getLogger(__name__)

def test_core_prepare_pipeline(tmp_path):
    # synthetic (docs x metrics)
    N, M = 200, 32
    rng = np.random.default_rng(0)
    X = np.clip(rng.normal(0.5, 0.25, size=(N, M)), 0, None).astype(np.float32)

    out_path = str(tmp_path / "core_pipeline.vpm.png")

    stages = [
        {"stage": "normalizer/normalize.NormalizeStage", "params": {"metric_names": [f"m{i}" for i in range(M)]}},
        {"stage": "amplifier/feature_engineer.FeatureEngineerStage", "params": {"nonlinearity_hint": None}},
        {"stage": "organizer/organize.Organize", "params": {"sql_query": ""}},  # identity sort
        {"stage": "vpm/write.VPMWrite", "params": {"output_path": out_path}},
    ]

    result, ctx = PipelineExecutor(stages).run(X, context={"enable_gif": False})

    logger.debug(f"Pipeline result shape: {result.shape}")
    logger.debug(f"Pipeline context: {ctx}")
    logger.info(f"VPM image written to: {out_path}")

    # assertions
    assert result.shape == X.shape  # stages preserve matrix shape
    assert ctx["final_stats"]["vpm_shape"] == result.shape
  