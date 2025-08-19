# zeromodel/pipeline/executor.py
"""
Pipeline executor for ZeroModel with GIF logging integration.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage
from zeromodel.pipeline.utils.gif_metrics import _gif_metrics
from zeromodel.pipeline.utils.vpm_preview import _vpm_preview_uint8
from zeromodel.tools.gif_logger import GifLogger

from PIL import Image, ImageDraw, ImageFont
import io

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """
    Execute a sequence of pipeline stages on VPMs with optional GIF logging.

    This implements ZeroModel's "infinite memory" principle:
    "The answer is already here."
    """

    def __init__(self, stages: List[Dict[str, Any]]):
        self.stages = stages
        logger.info(f"PipelineExecutor initialized with {len(stages)} stages")

    def _load_stage(self, stage_path: str, params: Dict[str, Any]):
        """Load a pipeline stage from its path."""
        try:
            if "." in stage_path:
                pkg, clsname = stage_path.rsplit(".", 1)
            else:
                pkg, clsname = stage_path, "Stage"

            module_path = f"zeromodel.pipeline.{pkg.replace('/', '.')}"
            module = __import__(module_path, fromlist=[clsname])

            if hasattr(module, clsname):
                cls = getattr(module, clsname)
            else:
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, PipelineStage)
                        and attr != PipelineStage
                    ):
                        cls = attr
                        break
                else:
                    raise ImportError(f"No PipelineStage class found in {module_path}")

            return cls(**params)

        except Exception as e:
            logger.error(f"Failed to load stage {stage_path}: {e}")
                    # Return a no-op stage that passes through data
            msg = f"Failed to load stage {stage_path}: {e}"
            class NoOpStage(PipelineStage):
                name = stage_path
                category = "error"
                
                def validate_params(self): 
                    pass
                
                def process(self, vpm, context=None):
                    return vpm, {"error": msg, "stage": stage_path}

            return NoOpStage(**params)
        
    def _init_context(self, context: Dict[str, Any] | None) -> Dict[str, Any]:
        """Initialize context with required fields."""
        ctx = {} if context is None else dict(context)
        
        # Initialize provenance with pipeline start
        if "provenance" not in ctx:
            ctx["provenance"] = []
            self._record(ctx, kind="pipeline_start", timestamp=np.datetime64("now"))
        
        if "pipeline_start_time" not in ctx:
            ctx["pipeline_start_time"] = np.datetime64("now")
        
        if "stats" not in ctx:
            ctx["stats"] = {}

        ctx = {
            "enable_gif": True,
            "gif_scale": 4,
            "gif_fps": 6,
           **ctx
        }

        
        return ctx

    def _record(self, ctx: Dict[str, Any], **event):
        """Record event in provenance with timestamp."""
        ctx["provenance"].append({
            "timestamp": np.datetime64("now"),
            **event
        })




    def run(
        self, vpm: np.ndarray, context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute the pipeline on a VPM with optional GIF logging.

        Args:
            vpm: Input VPM as numpy array
            context: Optional context dictionary with GIF logging configuration

        Returns:
            (processed_vpm, final_context) - Enhanced VPM and complete context
        """
        if context is None:
            context = {}

        ctx = self._init_context(context)
        cur = vpm

        logger.info(f"Executing pipeline with {len(self.stages)} stages")

        # Ensure gif logger lives in ctx
        gif = ctx.get("gif_logger")
        if gif is None and ctx.get("enable_gif", True):
            try:
                gif = GifLogger(
                    max_frames=ctx.get("gif_max_frames", 2000),
                    vpm_scale=ctx.get("gif_scale", 4),
                    strip_h=ctx.get("gif_strip_h", 40),
                )
                ctx["gif_logger"] = gif
            except Exception as _:
                gif = None

        # Add an initial frame before any stage runs
        if gif is not None:
            _gif_add(ctx, cur, tl_value=None, tag="start")

        for i, spec in enumerate(self.stages):
            stage_path = spec["stage"]
            params = spec.get("params", {})

            # stage start event
            self._record(
                ctx,
                kind="stage_start",
                stage=stage_path,
                index=i,
                params=params,
                input_shape=tuple(cur.shape),
            )
            t0 = time.time()

            try:
                logger.info(f"Executing stage {i + 1}/{len(self.stages)}: {stage_path}")
                stage = self._load_stage(stage_path, params)

                if ctx.get("gif_logger") is not None:
                    _gif_add(ctx, cur, tl_value=None, tag=f"pre:{i}")
                
                start_time = time.time()
                stage.validate_params()

                cur, meta = stage.process(cur, ctx)

                execution_time = time.time() - start_time

                self._record(
                    ctx,
                    kind="stage_end",
                    stage=stage_path,
                    index=i,
                    ok=True,
                    elapsed_sec=execution_time,
                    metadata=meta or {},
                )

                context[f"stage_{i}"] = {
                    "stage": stage_path,
                    "input_shape": tuple(vpm.shape),
                    "output_shape": tuple(cur.shape),
                    "elapsed_sec": execution_time,
                    "metadata": meta or {},
                }
                logger.info(f"Stage {i} metadata: {meta}")

                # AFTER: capture new state (include TL if provided by meta)
                tl_val = float(meta["tl_mass_avg"]) if isinstance(meta, dict) and "tl_mass_avg" in meta else None
                if ctx.get("gif_logger") is not None:
                    _gif_add(ctx, cur, tl_value=tl_val, tag=f"post:{i}")

            except Exception as e:
                dt = time.time() - t0

                logger.exception(f"Stage {stage_path} failed")
                # stage end (failure)
                self._record(
                    ctx,
                    kind="stage_end",
                    stage=stage_path,
                    index=i,
                    ok=False,
                    elapsed_sec=dt,
                    error=str(e),
                )

                context[f"stage_{i}_error"] = {
                    "stage": stage_path,
                    "error": str(e),
                    "timestamp": np.datetime64("now"),
                }
                # Continue with current VPM (don't break the pipeline)

        _gif_capture(ctx, cur, label="final", per_slice=True)  # emit T frames

        # Add final statistics
        context["final_stats"] = {
            "vpm_shape": cur.shape,
            "vpm_min": float(cur.min()),
            "vpm_max": float(cur.max()),
            "vpm_mean": float(cur.mean()),
            "pipeline_stages": len(self.stages),
            "total_execution_time": sum(
                context.get(f"stage_{i}", {}).get("elapsed_sec", 0.0)
                for i in range(len(self.stages))
            ),
        }

        # Save GIF if requested
        if gif is not None and context.get("gif_path"):
            out_path = context.get("gif_path")
            fps = context.get("gif_fps", 6)
            try:
                gif.save_gif(out_path, fps=fps, optimize=True, loop=0)
                context["gif_saved"] = out_path
                logger.info(f"GIF saved to {out_path}")
            except Exception as e:
                logger.exception(f"Failed to save GIF: {e}")
                context["gif_error"] = str(e)

        return cur, context

    def _get_initialized_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize context with required fields."""
        if context is None:
            context = {}

        if "provenance" not in context:
            context['provenance'] = []

        if 'pipeline_start_time' not in context:
            context['pipeline_start_time'] = np.datetime64('now')

        return context
    
def _vpm_to_uint8_preview(vpm_slice: np.ndarray) -> np.ndarray:
    v = vpm_slice.astype(np.float32)
    lo, hi = np.percentile(v, 1.0), np.percentile(v, 99.0)
    if hi <= lo: hi = lo + 1e-6
    y = np.clip((v - lo) / (hi - lo), 0, 1)
    r = (y**0.9) * 255.0; g = (y**0.8) * 255.0; b = (y**0.7) * 255.0
    return np.stack([r, g, b], axis=-1).astype(np.uint8)

def _gif_capture(ctx: dict, vpm: np.ndarray, label: str = "", per_slice: bool = False):
    gif = ctx.get("gif_logger")
    if gif is None:
        return

    if not per_slice or vpm.ndim == 2:
        frame = _frame_from_context_or_vpm(ctx, vpm if vpm.ndim == 2 else vpm[0])
        step = _gif_next_step(ctx)
        gif.add_frame(
            frame,
            {"step": step, "loss": float("nan"), "val_loss": float("nan"),
             "acc": float(np.mean(frame)/255.0), "alerts": {"tag": label}}
        )
        return

    # Per-slice frames
    T = vpm.shape[0]
    for t in range(T):
        frame = _frame_from_context_or_vpm(ctx, vpm[t])
        step = _gif_next_step(ctx)
        gif.add_frame(
            frame,
            {"step": step, "loss": float("nan"), "val_loss": float("nan"),
             "acc": float(np.mean(frame)/255.0), "alerts": {"tag": f"{label}:{t}"}}
        )

def _frame_from_context_or_vpm(ctx: dict, vpm: np.ndarray) -> np.ndarray:
    arr = ctx.get("vpm_uint8")
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] == 3:
        return _apply_debug_stripe(arr.astype(np.uint8, copy=False), ctx)

    png_bytes = ctx.get("vpm_png_bytes")
    if png_bytes:
        try:
            im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            return _apply_debug_stripe(np.array(im, dtype=np.uint8), ctx)
        except Exception:
            pass

    # fallback to your existing preview
    frame = _vpm_preview_uint8(vpm)
    return _apply_debug_stripe(frame, ctx)


def _gif_next_step(ctx: dict) -> int:
    step = ctx.setdefault("_gif_step", 0)
    ctx["_gif_step"] = step + 1
    return step


def _gif_add(ctx: dict, vpm: np.ndarray, tl_value: float | None = None, tag: str = ""):
    gif = ctx.get("gif_logger")
    if gif is None:
        return
    frame = _frame_from_context_or_vpm(ctx, vpm)
    step = _gif_next_step(ctx)
    gif.add_frame(frame, _gif_metrics(step=step, vpm=vpm, tl_value=tl_value, tag=tag))

def _apply_debug_stripe(frame: np.ndarray, ctx: dict) -> np.ndarray:
    """
    If ctx['gif_debug_stripe'] is True, overlay a visible debug stripe
    at the top of the frame so the stripe is unmistakable in GIFs.
    """
    if not ctx.get("gif_debug_stripe", False):
        return frame

    # Config with defaults
    stripe_h = int(ctx.get("gif_debug_stripe_height", 16))
    color    = tuple(ctx.get("gif_debug_stripe_color", (255, 0, 255)))  # magenta
    text_on  = bool(ctx.get("gif_debug_stripe_text", True))
    text     = str(ctx.get("gif_debug_stripe_label", "DEBUG STRIPE"))

    H, W, _ = frame.shape
    stripe_h = max(2, min(stripe_h, max(8, H // 6)))

    im = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(im)

    # top bar
    draw.rectangle([(0, 0), (W - 1, stripe_h - 1)], fill=color)

    # thin black separator under stripe (makes it pop)
    draw.line([(0, stripe_h), (W - 1, stripe_h)], fill=(0, 0, 0), width=1)

    # optional text
    if text_on:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((4, 2), text, fill=(0, 0, 0), font=font)

    return np.asarray(im, dtype=np.uint8)
