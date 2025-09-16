# zeromodel/tools/gif_logger.py

from __future__ import annotations
import time
import numpy as np
from PIL import Image, ImageDraw

class GifLogger:
    def __init__(self, max_frames=2000, vpm_scale=4, strip_h=40, bg=(10, 10, 12), fps: int | None = None, **kwargs):
        self.frames = []
        self.meta = []  # store dicts of metrics for bottom strip
        self.max_frames = int(max_frames)
        self.vpm_scale = int(vpm_scale)
        self.strip_h = int(strip_h)
        self.bg = tuple(bg) if isinstance(bg, (list, tuple)) else (10, 10, 12)
        self.default_fps = int(fps) if fps is not None else 6  # default 6 if not provided

    def add_frame(self, vpm_uint8: np.ndarray, metrics: dict):
        """
        vpm_uint8: HxWx3 (uint8) — small VPM or tile at this timestep
        metrics:   {"step": int, "loss": float, "val_loss": float, "acc": float, "alerts": dict}
        """
        if len(self.frames) >= self.max_frames:
            return  # cheap backpressure; or implement decimation

        # ✅ make robust to 2-D, 1-ch, 4-ch, float inputs
        vpm_uint8 = self._ensure_rgb_uint8(vpm_uint8)

        self.frames.append(vpm_uint8.copy())
        self.meta.append(
            {
                "t": time.time(),
                "step": metrics.get("step", len(self.frames) - 1),
                "loss": float(metrics.get("loss", np.nan)),
                "val_loss": float(metrics.get("val_loss", np.nan)),
                "acc": float(metrics.get("acc", np.nan)),
                "alerts": dict(metrics.get("alerts", {})),
            }
        )

    def _compose_panel(self, vpm: np.ndarray, hist: list) -> Image.Image:
        """
        Build one composite frame:
        - top: scaled VPM (nearest)
        - bottom: metric mini-timeline (last K points)
        """
        vpm = self._ensure_rgb_uint8(vpm)

        # --- top: scale the VPM ---
        H, W, _ = vpm.shape
        scale = self.vpm_scale
        top = Image.fromarray(vpm).resize(
            (W * scale, H * scale), resample=Image.NEAREST
        )

        # --- bottom: timeline strip ---
        K = min(300, len(hist))
        tail = hist[-K:]
        strip_w = top.width
        strip = Image.new("RGB", (strip_w, self.strip_h), self.bg)
        draw = ImageDraw.Draw(strip)

        # Normalize and plot tiny sparklines
        def norm(series):
            arr = np.array(series, dtype=np.float32)
            good = np.isfinite(arr)
            if good.sum() < 2:  # fallback
                return np.zeros_like(arr)
            a = arr[good]
            lo, hi = np.percentile(a, 5), np.percentile(a, 95)
            if hi - lo < 1e-8:
                hi = lo + 1e-8
            arr = np.clip((arr - lo) / (hi - lo), 0, 1)
            arr[~good] = np.nan
            return arr

        losses = norm([d["loss"] for d in tail])
        vlosses = norm([d["val_loss"] for d in tail])
        accs = norm([d["acc"] for d in tail])

        # helper to draw one sparkline
        def draw_line(vals, y0, color):
            if len(vals) < 2:
                return
            w = strip_w
            xs = np.linspace(0, w - 1, num=len(vals))
            pts = []
            for x, v in zip(xs, vals):
                if np.isnan(v):
                    continue
                y = int(y0 + (1.0 - v) * (self.strip_h / 3 - 6))
                pts.append((int(x), y))
            if len(pts) > 1:
                draw.line(pts, fill=color, width=1)

        # three stacked sparklines
        h3 = self.strip_h // 3
        draw_line(losses, 1 + 0 * h3, (200, 120, 120))  # loss
        draw_line(vlosses, 1 + 1 * h3, (120, 180, 220))  # val_loss
        draw_line(accs, 1 + 2 * h3, (140, 220, 140))  # acc

        # alert ticks (e.g., overfit) across bottom
        for i, d in enumerate(tail):
            x = int(i * (strip_w - 1) / max(1, K - 1))
            alerts = d["alerts"]
            if alerts.get("overfit", False):
                draw.line(
                    [(x, self.strip_h - 6), (x, self.strip_h - 1)],
                    fill=(255, 80, 80),
                    width=1,
                )
            if alerts.get("drift", False):
                draw.line(
                    [(x, self.strip_h - 12), (x, self.strip_h - 7)],
                    fill=(255, 200, 80),
                    width=1,
                )

        # --- stack top + bottom ---
        panel = Image.new("RGB", (top.width, top.height + strip.height), self.bg)
        panel.paste(top, (0, 0))
        panel.paste(strip, (0, top.height))
        return panel

    def save_gif(
        self,
        path: str = "training_heartbeat.gif",
        fps: int | None = None,
        optimize: bool = True,
        loop: int = 0,
        use_palette: bool = True,
        context: dict | None = None,
    ):
        """
        Save accumulated frames to an animated GIF.
        If fps is None, uses the fps provided at construction (default_fps).
        Args:
            path: Output file path.
            fps: Frames per second.
            optimize: Whether to optimize GIF size.
            loop: Number of loops (0 = infinite).
            use_palette: If True, quantize to 256-color palette (smaller, lossy).
                        If False, keep RGB frames (bigger, lossless).
        """

        if not self.frames:
            raise RuntimeError("No frames added.")

        # Compose panels (stride decimation if too many frames)
        panels = []
        stride = max(1, len(self.frames) // self.max_frames)
        for i in range(0, len(self.frames), stride):
            panels.append(self._compose_panel(self.frames[i], self.meta[: i + 1]))

        # Auto override: analysis mode forces RGB
        if context and context.get("enable_analysis", False):
            use_palette = False

        if use_palette:
            # Convert to palette (lossy, but smaller)
            frames_out = [im.convert("P", palette=Image.ADAPTIVE, colors=256) for im in panels]
        else:
            # Keep RGB (full fidelity, bigger file)
            frames_out = [im.convert("RGB") for im in panels]

        eff_fps = int(self.default_fps if fps is None else fps)
        duration_ms = int(1000 / max(1, eff_fps))

        # Pillow's GIF writer supports optimize/disposal; keep them.
        frames_out[0].save(
            path,
            save_all=True,
            append_images=frames_out[1:],
            duration=duration_ms,
            loop=loop,
            optimize=optimize,
            disposal=2,
        )
        return path

    def _ensure_rgb_uint8(self, img) -> np.ndarray:
        """
        Normalize various image-like inputs to (H, W, 3) uint8 RGB.

        Accepts:
          - PIL.Image.Image (any mode)
          - np.ndarray with shapes: (H,W), (H,W,1), (H,W,3), (H,W,4)
        Dtypes supported: uint8, float (scaled), int (clipped).

        Rules:
          - RGBA is alpha-composited over self.bg
          - Grayscale / single-channel is replicated to RGB
          - Float arrays in [0,1] → *255; in [0,255] → clipped; otherwise percentile-normalized
          - NaN/Inf are treated as 0
        """
        # --- 1) Convert PIL → np array ---
        if isinstance(img, Image.Image):
            if img.mode == "RGBA":
                # composite on background
                bg = Image.new("RGB", img.size, self.bg)
                img = Image.alpha_composite(bg.convert("RGBA"), img).convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
            return arr

        # --- 2) Must be ndarray from here ---
        arr = np.asarray(img)
        if arr.ndim not in (2, 3):
            raise ValueError(f"_ensure_rgb_uint8: expected 2D or 3D array, got shape {arr.shape}")

        # --- 3) Handle dtype → float32 pipeline, clean NaNs/Inf ---
        if arr.dtype != np.uint8:
            x = arr.astype(np.float32, copy=False)
            # Replace NaNs/Infs
            bad = ~np.isfinite(x)
            if bad.any():
                x = x.copy()
                x[bad] = 0.0
        else:
            x = arr

        # --- 4) Channel handling ---
        if x.ndim == 2:                     # (H, W) → gray
            x = x[:, :, None]               # (H, W, 1)
        C = x.shape[-1]
        if C == 1:
            x = np.repeat(x, 3, axis=-1)    # gray → RGB
        elif C == 3:
            pass                            # already RGB-like
        elif C == 4:
            # RGBA → composite on bg
            if x.dtype != np.uint8:
                # Bring to 0..255 first
                x = self._to_uint8(x)
            rgb = x[..., :3].astype(np.float32)
            a = (x[..., 3:4].astype(np.float32) / 255.0)
            bg = np.array(self.bg, dtype=np.float32).reshape(1, 1, 3)
            x = (rgb * a + bg * (1.0 - a)).astype(np.uint8)
        else:
            raise ValueError(f"_ensure_rgb_uint8: unsupported channel count {C}")

        # --- 5) Ensure uint8 range ---
        if x.dtype != np.uint8:
            x = self._to_uint8(x)

        # Safety clamp & contiguous
        x = np.clip(x, 0, 255).astype(np.uint8, copy=False)
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"_ensure_rgb_uint8: final shape invalid {x.shape}")
        return x

    def _to_uint8(self, x: np.ndarray) -> np.ndarray:
        """
        Map numeric array to uint8 safely:
          - If 0..1 → *255
          - Else if 0..255 → clip
          - Else percentile (1,99) normalization
        """
        x = x.astype(np.float32, copy=False)
        # guard NaN/Inf
        bad = ~np.isfinite(x)
        if bad.any():
            x = x.copy()
            x[bad] = 0.0

        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))

        if xmax <= 1.0 + 1e-6 and xmin >= 0.0:
            y = x * 255.0
        elif xmax <= 255.0 + 1e-6 and xmin >= 0.0:
            y = x
        else:
            # robust normalization (like your preview helpers)
            lo, hi = np.percentile(x, (1.0, 99.0))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = xmin, max(xmin + 1e-6, xmax)
            y = (x - lo) / (hi - lo)
            y = np.clip(y, 0.0, 1.0) * 255.0

        return y.astype(np.uint8, copy=False)
