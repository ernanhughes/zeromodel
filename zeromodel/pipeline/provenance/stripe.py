#  zeromodel/pipeline/provenance/stripe.py
from __future__ import annotations

import io
import hashlib
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from zeromodel.pipeline.base import PipelineStage


class Stripe(PipelineStage):
    """
    Append a bottom stripe with text (e.g., hash, stage count, timestamp).
    Consumes context['vpm_uint8'] or context['vpm_png_bytes'].
    Writes back updated 'vpm_uint8' and 'vpm_png_bytes'.

    Params
    ------
    height: int = 28
        Stripe height in pixels.
    bg: tuple[int,int,int] = (10,10,12)
        Background color of the stripe.
    fg: tuple[int,int,int] = (230,230,230)
        Foreground/text color.
    separator_px: int = 2
        Thickness of a high-contrast separator line above the stripe (0 to disable).
    separator_color: tuple[int,int,int] = (255,0,255)
        Color for the separator line (visible even on gray tiles).
    text: str|None
        Format string; supports {sha}, {stages}, {width}, {height}. If None, a default is used.
    font_name: str|None
        Optional PIL font name/path. Falls back to default PIL font if unavailable.
    font_size: int|None
        Optional font size. Ignored when using ImageFont.load_default().
    left_pad: int = 6
        Left padding for the text in the stripe.
    top_pad: int = 6
        Top padding for the text in the stripe.
    """

    name = "add_stripe"
    category = "visual"

    def __init__(self, **params):
        super().__init__(**params)
        self.height: int = int(params.get("height", 28))
        self.bg = tuple(params.get("bg", (10, 10, 12)))
        self.fg = tuple(params.get("fg", (230, 230, 230)))

        self.separator_px: int = int(params.get("separator_px", 2))
        self.separator_color = tuple(params.get("separator_color", (255, 0, 255)))

        self.text = params.get("text", None)
        self.font_name = params.get("font_name", None)
        self.font_size = params.get("font_size", None)
        self.left_pad: int = int(params.get("left_pad", 6))
        self.top_pad: int = int(params.get("top_pad", 6))

    # Back-compat alias if base uses validate_params()
    def validate_params(self):
        assert self.height >= 0
        assert self.separator_px >= 0
        assert self.left_pad >= 0 and self.top_pad >= 0

    # Back-compat alias for base helpers (some repos used get_context/record_provenance names)
    def get_context(self, context: Dict[str, Any] | None) -> Dict[str, Any]:
        if hasattr(super(), "get_context"):
            return super().get_context(context)  # type: ignore[attr-defined]
        # fallback to older base api
        if context is None:
            context = {}
        context.setdefault("provenance", [])
        return context

    def record_provenance(self, context: Dict[str, Any], stage_name: str, params: Dict[str, Any]):
        if hasattr(super(), "record_provenance"):
            return super().record_provenance(context, stage_name, params)  # type: ignore[attr-defined]
        context.setdefault("provenance", []).append(
            {"stage": stage_name, "params": params, "timestamp": np.datetime64("now")}
        )

    # ---- helpers ----
    def _ensure_rgb(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return arr.astype(np.uint8, copy=False)
        raise ValueError(f"Unsupported vpm_uint8 shape: {arr.shape}")

    def _load_font(self):
        # Try user-specified font; otherwise default
        if self.font_name:
            try:
                size = int(self.font_size or 12)
                return ImageFont.truetype(self.font_name, size=size)
            except Exception:
                pass
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    def _compute_sha12(self, context: Dict[str, Any], arr_rgb: np.ndarray | None, base_png: bytes | None) -> str:
        # Prefer existing provenance hash (if already computed upstream)
        sha = context.get("provenance_sha256")
        if isinstance(sha, str) and sha:
            return sha[:12]

        # Otherwise compute from available artifact: PNG → fastest/stable
        if base_png:
            h = hashlib.sha256(base_png).hexdigest()
            context["provenance_sha256"] = h
            return h[:12]

        if arr_rgb is not None:
            h = hashlib.sha256(arr_rgb.tobytes()).hexdigest()
            context["provenance_sha256"] = h
            return h[:12]

        return ""  # nothing available

    # ---- core ----
    def process(self, vpm, context: Dict[str, Any] | None = None) -> Tuple[Any, Dict[str, Any]]:
        context = self.get_context(context)

        # Pull base image from context: prefer vpm_uint8, else vpm_png_bytes
        arr = context.get("vpm_uint8")
        png_bytes: bytes | None = context.get("vpm_png_bytes")

        if arr is None and png_bytes:
            base = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            arr = np.array(base, dtype=np.uint8)

        if arr is None:
            # Nothing to draw on; no-op
            return vpm, {"status": "no_base_image"}

        arr = self._ensure_rgb(arr)
        H, W, _ = arr.shape

        # Separator (drawn on top of the base before stripe)
        panel_h = H + (self.height if self.height > 0 else 0)
        panel = Image.new("RGB", (W, panel_h), (0, 0, 0))
        panel.paste(Image.fromarray(arr), (0, 0))

        if self.separator_px > 0 and self.height > 0:
            draw = ImageDraw.Draw(panel)
            y0 = max(0, H - self.separator_px)
            y1 = H - 1
            draw.rectangle([0, y0, W - 1, y1], fill=self.separator_color)

        # Stripe
        if self.height > 0:
            stripe = Image.new("RGB", (W, self.height), self.bg)
            draw_s = ImageDraw.Draw(stripe)

            # Text template
            sha12 = self._compute_sha12(context, arr, png_bytes)
            stages = len(context.get("provenance", []))
            txt = self.text or "sha={sha} | stages={stages} | {width}x{height}"
            txt = txt.format(sha=sha12, stages=stages, width=W, height=H)

            # Font + draw
            font = self._load_font()
            # Optional subtle shadow for readability
            if font is not None:
                shadow = (0, 0, 0)
                draw_s.text((self.left_pad + 1, self.top_pad + 1), txt, fill=shadow, font=font)
            draw_s.text((self.left_pad, self.top_pad), txt, fill=self.fg, font=font)

            panel.paste(stripe, (0, H))

        # Update context artifacts
        panel_np = np.array(panel, dtype=np.uint8)
        context["vpm_uint8"] = panel_np
        buf = io.BytesIO()
        Image.fromarray(panel_np).save(buf, format="PNG", optimize=True)
        context["vpm_png_bytes"] = buf.getvalue()


        # provenance note
        self.record_provenance(
            context,
            self.name,
            {
                "height": self.height,
                "bg": self.bg,
                "fg": self.fg,
                "separator_px": self.separator_px,
                "separator_color": self.separator_color,
                "text_applied": bool(self.height > 0),
            },
        )

        return vpm, {
            "stripe_height": self.height,
            "input_wh": (W, H),
            "output_wh": (W, panel_h),
            "text": txt if self.height > 0 else "",
        }
