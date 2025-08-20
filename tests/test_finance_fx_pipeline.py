# tests/test_finance_fx_pipeline.py
import logging
import os
import numpy as np
import pandas as pd

from zeromodel.pipeline.executor import PipelineExecutor

logger = logging.getLogger(__name__)

def test_finance_fx_pipeline(tmp_path):
    # --- Step 1. Generate synthetic FX (EUR/USD) data ---
    N = 500
    rng = np.random.default_rng(42)

    # Price series as geometric random walk
    returns = rng.normal(0, 0.001, size=N)
    price = 1.10 + np.cumsum(returns)
    volume = rng.integers(100, 1000, size=N)

    df = pd.DataFrame({
        "Price": price,
        "Volume": volume,
        "Return": np.r_[0, np.diff(price)],
        "MA5": pd.Series(price).rolling(5).mean().fillna(method="bfill"),
        "MA20": pd.Series(price).rolling(20).mean().fillna(method="bfill"),
        "Volatility": pd.Series(returns).rolling(10).std().fillna(0)
    })

    # Normalize (z-score per column)
    X = (df - df.mean()) / (df.std() + 1e-9)
    X = X.to_numpy().astype(np.float32)

    # --- Step 2. Define pipeline ---
    out_gif  = str(tmp_path / "fx_hier.gif")
    out_img  = str(tmp_path / "fx_hier.png")

    stages = [
        {"stage": "normalizer/normalize.NormalizeStage", "params": {}},
        {"stage": "organizer/hierarchical_view.HierarchicalView", "params": {
            "levels": 4,
            "row_frac": 0.5, "col_frac": 0.5,
            "window_mode": "quadrants",
            "norm_mode": "local_subset",
            "alpha": 0.97,
            "beam_size": 1,
            "early_stop_epsilon": 1e-3,
            "top_left_params": {
                "metric_mode": "mean",
                "iterations": 6,
                "monotone_push": True,
                "reverse": True,
                "clip_percent": 0.005,
                "stretch": True,
                "push_corner": "tl"
            }
        }},
        {"stage": "vpm/write.VPMWrite", "params": {"output_path": out_img}},
    ]

    context = {
        "enable_gif": True,
        "gif_path": out_gif,
        "gif_scale": 4,
        "gif_fps": 6,
    }

    # --- Step 3. Run pipeline ---
    result, ctx = PipelineExecutor(stages).run(X, context=context)

    # --- Step 4. Assertions ---
    assert ctx.get("gif_saved") == out_gif
    assert os.path.exists(out_gif), "GIF not created"
    assert os.path.getsize(out_gif) > 0, "GIF is empty"

    logger.info(f"FX pipeline result shape: {result.shape}")
    logger.info(f"Hierarchical GIF written to: {out_gif}")
    logger.info(f"Static PNG written to: {out_img}")
