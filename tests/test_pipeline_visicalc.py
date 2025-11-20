import logging

import numpy as np

from zeromodel.pipeline.executor import PipelineExecutor
from zeromodel.pipeline.explainability.visicalc import format_visicalc_stats

logger = logging.getLogger(__name__)


def test_visicalc_stage_basic():
    # synthetic (docs x metrics)
    N, M = 96, 12
    rng = np.random.default_rng(42)
    X = np.clip(rng.normal(0.5, 0.2, size=(N, M)), 0.0, 1.0).astype(np.float32)

    metric_names = [f"m{i}" for i in range(M)]

    stages = [
        {
            "stage": "normalize/normalize.NormalizeStage",
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
                "frontier_metric_index": 5,
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

    # This is how PipelineExecutor exposes per-stage metadata:
    stats = ctx["stage_3"]["metadata"]
    logger.info(format_visicalc_stats(stats))

    # core invariants
    assert "visicalc" in stats
    vstats = stats["visicalc"]

    assert vstats["shape"] == result.shape
    assert "global" in vstats
    assert "frontier" in vstats
    assert "row_regions" in vstats
    assert "per_metric" in vstats
    assert len(vstats["feature_names"]) == len(vstats["feature_vector"])
  

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
                "frontier_metric_index": 2,
                "row_region_splits": 4,  # ask for 4 regions
            },
        },
    ]

    result, ctx = PipelineExecutor(stages).run(X, context={})

    assert result.shape == X.shape

    stats = ctx["stage_1"]["metadata"]
    logger.info(format_visicalc_stats(stats))

    vstats = stats["visicalc"]
    row_regions = vstats["row_regions"]

    # confirm we got 4 regions, and they cover [0, N)
    assert len(row_regions) == 4
    bounds = sorted(
        (v["row_start"], v["row_end"]) for v in row_regions.values()
    )
    assert bounds[0][0] == 0
    assert bounds[-1][1] == N
