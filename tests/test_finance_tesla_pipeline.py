# tests/test_finance_tesla_pipeline.py
import logging
import os
import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from zeromodel.pipeline.executor import PipelineExecutor

logger = logging.getLogger(__name__)

@pytest.mark.slow
def test_finance_tesla_pipeline(tmp_path):
    # --- Step 1. Download Tesla 1-min bars (past 1 day) ---
    df = yf.download("TSLA", interval="1m", period="1d")
    assert not df.empty, "No data downloaded from yfinance"

    # --- Step 2. Engineer features ---
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Volatility"] = df["Return"].rolling(5).std().fillna(0)
    df["MA5"] = df["Close"].rolling(5).mean().fillna(method="bfill")
    df["MA20"] = df["Close"].rolling(20).mean().fillna(method="bfill")

    # Normalize (z-score per column)
    X = (df - df.mean()) / (df.std() + 1e-9)
    X = X.to_numpy().astype(np.float32)

    # --- Step 3. Define pipeline ---
    out_gif  = str(tmp_path / "tesla_hier.gif")
    out_img  = str(tmp_path / "tesla_hier.png")

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

    # --- Step 4. Run pipeline ---
    result, ctx = PipelineExecutor(stages).run(X, context=context)

    # --- Step 5. Assertions ---
    assert result.ndim == 2
    assert result.shape[0] > 0 and result.shape[1] > 0
    assert os.path.exists(out_gif) and os.path.getsize(out_gif) > 0
    assert ctx.get("gif_saved") == out_gif
    assert os.path.exists(out_gif), "GIF not created"
    assert os.path.getsize(out_gif) > 0, "GIF is empty"

    logger.info(f"Tesla pipeline result shape: {result.shape}")
    logger.info(f"Hierarchical GIF written to: {out_gif}")
    logger.info(f"Static PNG written to: {out_img}")
