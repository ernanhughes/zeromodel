# tests/test_core_pipeline_prepare.py
import logging
import os

import numpy as np

from zeromodel.pipeline.executor import PipelineExecutor

logger = logging.getLogger(__name__)

def test_core_prepare_pipeline(tmp_path):
    # synthetic (docs x metrics)
    N, M = 200, 32
    rng = np.random.default_rng(0)
    X = np.clip(rng.normal(0.5, 0.25, size=(N, M)), 0, None).astype(np.float32)

    out_path = str(tmp_path / "core_pipeline.vpm.png")
    out_gif  = str(tmp_path / "core_pipeline_heartbeat.gif")

    stages = [
        {"stage": "normalizer/normalize.NormalizeStage", "params": {"metric_names": [f"m{i}" for i in range(M)]}},
        {"stage": "amplifier/feature_engineer.FeatureEngineerStage", "params": {"nonlinearity_hint": None}},
        {"stage": "organizer/organize.Organize", "params": {"sql_query": ""}},  # identity sort
        {"stage": "organizer/top_left.TopLeft No he's good he's good too",
          "params": {
          "metric_mode": "mean",       # for 2D, 'mean' is fine; for RGB use 'luminance'
          "iterations": 12,            # more alternating sorts
          "monotone_push": True,       # OK cumulative “smear” into TL
          "stretch": True,             # contrast stretch
          "clip_percent": 0.0,         # keep full dynamic range (or 0.005)
          "reverse": True             # bright → top/left
          }},
        {"stage": "vpm/write.VPMWrite", "params": {"output_path": out_path}},
    ]

    context = {
        "enable_gif": True,            # turn on frame capture in the executor
        "gif_path": out_gif,           # where to save the animation
        "gif_scale": 4,                # visual scale of the VPM pane
        "gif_fps": 6,                  # playback speed
        # optional:
        # "gif_max_frames": 2000,
        # "gif_strip_h": 40,
    }

    result, ctx = PipelineExecutor(stages).run(X, context=context)

    from zeromodel.vpm.stdm import top_left_mass

    Kc, Kr, alpha = 16, 80, 0.97
    tl_before = float(top_left_mass(X, Kr=Kr, Kc=Kc, alpha=alpha))
    tl_after  = float(top_left_mass(result, Kr=Kr, Kc=Kc, alpha=alpha))
    logger.info(f"TL before={tl_before:.4f}  TL after={tl_after:.4f}  gain={(tl_after-tl_before)/max(1e-9,tl_before)*100:.2f}%")


    logger.debug(f"Pipeline result shape: {result.shape}")
    logger.debug(f"Pipeline context: {ctx}")
    logger.info(f"VPM image written to: {out_path}")
    logger.info(f"GIF written to: {ctx.get('gif_saved')}")

    # assertions
    assert result.shape == X.shape  # stages preserve matrix shape
    assert ctx["final_stats"]["vpm_shape"] == result.shape
  
    assert ctx.get("gif_saved") == out_gif
    assert os.path.exists(out_gif), "GIF file was not created"
    assert os.path.getsize(out_gif) > 0, "GIF file is empty"