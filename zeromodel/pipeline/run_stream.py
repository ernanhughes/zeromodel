#  zeromodel/pipeline/run_stream.py
from __future__ import annotations
import numpy as np
from zeromodel.maestro.online import OnlineMaestro
from zeromodel.tools.overlay import overlay_residual   # simple heatmap blend
from zeromodel.tools.gif_logger import GifLogger          # from earlier section
from zeromodel.maestro.pgw_norm import pgw_normalize # group-wise
# group_slices comes from config

def run_stream(source, online_maestro: OnlineMaestro, gif_out, group_slices):
    logger = GifLogger(gif_out, fps=8)
    for evt in source:
        frame = evt["frame"]             # (H,W,C) float32 [0,1]
        # (optional) per-frame group-wise target normalization for visualization
        norm = pgw_normalize(frame, group_slices)
        res_map, phase = online_maestro.step(norm)   # residual + phase logits
        blended = overlay_residual(frame, res_map)   # RGB blend
        text = f"step {evt['step']}  phase {phase.argmax().item()}"
        logger.add(blended, text=text)
    logger.save()
