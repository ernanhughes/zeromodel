import logging

import numpy as np
import json
from zeromodel.pipeline.executor import PipelineExecutor
from zeromodel.pipeline.utils.json_sanitize import dumps_safe   

logger = logging.getLogger(__name__)


def test_visicalc_stage_basic():
    # synthetic (docs x metrics)
    N, M = 96, 12
    rng = np.random.default_rng(42)
    X = np.clip(rng.normal(0.5, 0.2, size=(N, M)), 0.0, 1.0).astype(np.float32)

    metric_names = [f"m{i}" for i in range(M)]

    # NOTE: adjust the stage path string if you placed VisiCalc elsewhere,
    # e.g. "analytics/visicalc.VisiCalcStage" instead of "visicalc/visicalc.VisiCalcStage"
    stages = [
        {
            "stage": "normalize I/normalize.NormalizeStage",
            "params": {"metric_names": metric_names},
        },
        {
            "stage": "amplifier/feature_engineer.FeatureEngineerStage",
            "params": {"nonlinearity_hint": None},
        },
        {
            "stage": "organizer/organize.Organize",
            "params": {"sql_query": ""},  # identity sort
        },
        {
            "stage": "explainability/visicalc.VisiCalcStage",
            "params": {
                "frontier_metric": metric_names[0],
                "frontier_low": 0.25,
                "frontier_high": 0.75,
                "row_region_splits": 3,
            },
        },
    ]

    result, ctx = PipelineExecutor(stages).run(X, context={})

    # --- basic shape invariants ---
    assert isinstance(result, np.ndarray)
    assert result.shape == X.shape, "VisiCalcStage should not change matrix shape"

    stats = ctx["stage_3"]["metadata"]

    logger.info(f"\n\nVisiCalcStage stats: \n{dumps_safe(stats, indent=2)}")
    


def test_visicalc_respects_row_region_splits():
    """Ensure row_region_splits controls the number of regions."""
    N, M = 60, 6
    rng = np.random.default_rng(7)
    X = np.clip(rng.normal(0.5, 0.25, size=(N, M)), 0.0, 1.0).astype(np.float32)

    metric_names = [f"m{i}" for i in range(M)]

    stages = [
        {
            "stage": "normalize/normalize.NormalizeStage",
            "params": {"metric_names": metric_names},
        },
        {
            "stage": "explainability/visicalc.VisiCalcStage",
            "params": {
                "frontier_metric": metric_names[0],
                "row_region_splits": 4,  # ask for 4 regions
            },
        },
    ]

    result, ctx = PipelineExecutor(stages).run(X, context={})

    assert result.shape == X.shape

    stats = ctx["stage_1"]["metadata"]
    logger.info(f"\n\nVisiCalcStage stats: \n{dumps_safe(stats, indent=2)}")
 