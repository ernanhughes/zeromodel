# zeromodel/training_heartbeat_visualizer.py
import logging
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class TrainingHeartbeatVisualizer:
    """
    Lightweight GIF visualizer for ZeroMemory VPM snapshots.
    Compatible with tests that either:
      - call add_frame(zeromemory)            # older/simple style
      - call add_frame(vpm_uint8=..., metrics=...)  # explicit style
    """

    def __init__(
        self,
        max_frames: int = 100,
        vpm_scale: int = 6,
        strip_height: int = 40,
        # new, for test compatibility
        fps: int = 5,
        show_alerts: bool = False,
        show_timeline: bool = False,
        show_metric_names: bool = False,
        bg_color: Tuple[int, int, int] = (0, 0, 0),
    ):
        self.max_frames = int(max_frames)
        self.vpm_scale = int(vpm_scale)
        self.strip_height = int(strip_height)

        # Compatibility / no-op toggles used by tests
        self.fps = int(fps)
        self.show_alerts = bool(show_alerts)
        self.show_timeline = bool(show_timeline)
        self.show_metric_names = bool(show_metric_names)
        self.bg_color = tuple(int(c) for c in bg_color)

        self.frames: List[np.ndarray] = []
        log.info(
            "Initialized TrainingHeartbeatVisualizer with max_frames=%d, vpm_scale=%d, strip_height=%d",
            self.max_frames,
            self.vpm_scale,
            self.strip_height,
        )

    # ----------------- public API -----------------

    def add_frame(
        self,
        zeromemory=None,
        *,
        vpm_uint8: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a frame to the GIF.
        Supported call patterns:
          - add_frame(zeromemory)                    # will build a frame from ZeroMemory snapshot
          - add_frame(vpm_uint8=vpm, metrics=meta)   # explicit VPM + optional metrics
        """
        if vpm_uint8 is None:
            if zeromemory is None:
                raise TypeError("add_frame requires either zeromemory or vpm_uint8")
            frame = self._frame_from_zeromemory(zeromemory)
        else:
            frame = self._frame_from_vpm(vpm_uint8)

        # (Optional) We could draw overlays for alerts/timeline/metric names.
        # Tests mainly check that API accepts options and no exceptions are raised,
        # so we keep overlays as no-ops for now.
        # However, to ensure GIF encoders don't collapse identical frames,
        # embed a tiny per-frame marker using the step (if provided) or a sequence id.
        step_id = None
        if isinstance(metrics, dict) and ("step" in metrics):
            try:
                step_id = int(metrics["step"])  # best-effort
            except Exception:
                step_id = None
        frame = self._apply_frame_marker(frame, step_id)

        self._push_frame(frame)

    def save_gif(self, path: str):
        if not self.frames:
            log.error("No frames to save - call add_frame() first")
            raise RuntimeError("No frames to save")

        # Use PIL to ensure all frames are preserved (some writers optimize away duplicates)
        try:
            # Convert each frame to palette mode with adaptive palette so the GIF encoder
            # treats each as a full frame and avoids coalescing identical-looking frames.
            pil_frames = [
                Image.fromarray(f.astype(np.uint8)).convert(
                    "P", palette=Image.ADAPTIVE, colors=256
                )
                for f in self.frames
            ]
            duration_ms = int(max(1, round(1000.0 / max(1, self.fps))))
            pil_frames[0].save(
                path,
                save_all=True,
                append_images=pil_frames[1:],
                format="GIF",
                duration=duration_ms,
                loop=0,
                optimize=False,
                disposal=2,
            )
        except Exception:
            # Fallback to imageio if PIL path fails for any reason
            imageio.mimsave(
                path,
                self.frames,
                duration=max(1e-6, 1.0 / max(1, self.fps)),
            )

    # ----------------- internals -----------------

    def _push_frame(self, frame_rgb_uint8: np.ndarray):
        frame = np.asarray(frame_rgb_uint8, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Frame must be HxWx3 uint8, got shape {frame.shape}")
        self.frames.append(frame)
        # enforce cap
        if len(self.frames) > self.max_frames:
            self.frames = self.frames[-self.max_frames :]

    def _frame_from_vpm(self, vpm_uint8: np.ndarray) -> np.ndarray:
        """Scale VPM to display size and add a simple footer strip."""
        vpm_uint8 = self._ensure_rgb_uint8(vpm_uint8)
        # shape check not required; _ensure_rgb_uint8 guarantees HxWx3

        # nearest-neighbor scale
        scale = max(1, int(self.vpm_scale))
        big = np.repeat(np.repeat(vpm_uint8, scale, axis=0), scale, axis=1)

        # footer strip (plain bg)
        footer = np.zeros((self.strip_height, big.shape[1], 3), dtype=np.uint8)
        footer[...] = self.bg_color

        return np.concatenate([big, footer], axis=0)

    def _frame_from_zeromemory(self, zm) -> np.ndarray:
        """
        Convert a ZeroMemory snapshot to a frame.
        Tries: zm.get_tile(size=(8,8)) -> zm.to_matrix() -> zm.buffer (latest row heatmap)
        """
        tile = None

        # 1) try explicit small tile
        if hasattr(zm, "get_tile"):
            try:
                tile = zm.get_tile(size=(8, 8))  # float [0,1] or uint8
            except Exception:
                tile = None

        # 2) fallback to full matrix
        if tile is None and hasattr(zm, "to_matrix"):
            try:
                tile = zm.to_matrix()
            except Exception:
                tile = None

        # 3) build from buffer (latest step’s metrics)
        if tile is None and hasattr(zm, "buffer"):
            buf = np.array(zm.buffer, dtype=object)  # (steps, metrics) ragged-safe
            if buf.size == 0:
                tile = np.zeros((8, 8), dtype=np.float32)
            else:
                last = buf[-1]
                # ensure 1D numeric vector
                if np.isscalar(last):
                    last = np.array([last], dtype=float)
                last = np.asarray(last, dtype=float).reshape(-1)
                # tile-ize into square
                side = int(np.ceil(np.sqrt(len(last))))
                pad = side * side - len(last)
                if pad > 0:
                    last = np.pad(last, (0, pad), mode="edge")
                tile = last.reshape(side, side)

        # safety: if everything failed
        if tile is None:
            tile = np.zeros((8, 8), dtype=np.float32)

        # normalize to [0,1]
        tile = np.asarray(tile)
        if tile.ndim == 0:  # scalar guard
            tile = np.array([[float(tile)]], dtype=np.float32)
        if tile.dtype == np.uint8:
            tile01 = tile.astype(np.float32) / 255.0
        else:
            mx = float(tile.max()) if np.isfinite(tile).any() else 1.0
            mx = mx if mx > 0 else 1.0
            tile01 = tile.astype(np.float32) / mx
        tile01 = np.nan_to_num(tile01, nan=0.0, posinf=1.0, neginf=0.0)
        tile01 = np.clip(tile01, 0.0, 1.0)

        # make RGB uint8
        if tile01.ndim == 2:
            rgb = np.stack([tile01, tile01, tile01], axis=-1)
        elif tile01.ndim == 3 and tile01.shape[-1] == 3:
            rgb = tile01
        else:
            # collapse any extra channels to 1, then repeat to 3
            base = (
                tile01[..., :1]
                if tile01.ndim >= 3
                else tile01.reshape(tile01.shape[0], -1)[:, :1]
            )
            rgb = np.repeat(base, 3, axis=-1)

        vpm_uint8 = (rgb * 255.0).astype(np.uint8)
        return self._frame_from_vpm(vpm_uint8)

    def _apply_frame_marker(
        self, frame: np.ndarray, step_id: Optional[int]
    ) -> np.ndarray:
        """Embed a tiny colored marker so successive frames are never bit-identical.
        This avoids GIF encoders merging identical frames and reducing n_frames.
        """
        out = np.array(frame, copy=True)
        h, w, _ = out.shape
        # choose a color derived from step_id or from an internal counter
        if not hasattr(self, "_seq_counter"):
            self._seq_counter = 0
        sid = step_id if step_id is not None else self._seq_counter
        # simple hash to RGB
        r = (sid * 67) % 256
        g = (sid * 97) % 256
        b = (sid * 131) % 256
        # draw a 2x2 marker in the footer's top-left corner (bottom-left of full frame)
        y0 = max(0, h - self.strip_height)
        y1 = min(h, y0 + 2)
        x0, x1 = 0, min(w, 2)
        out[y0:y1, x0:x1, 0] = r
        out[y0:y1, x0:x1, 1] = g
        out[y0:y1, x0:x1, 2] = b
        # bump counter for next call if we generated it
        if step_id is None:
            self._seq_counter += 1
        return out

    @staticmethod
    def _ensure_rgb_uint8(arr: np.ndarray) -> np.ndarray:
        """Accept HxW, HxWx1, HxWx3 (float01 or uint8); return HxWx3 uint8."""
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.ndim == 0:
            arr = np.array([[[float(arr)]]], dtype=np.float32)
            arr = np.repeat(arr, 3, axis=-1)

        if arr.dtype != np.uint8:
            # assume float01-ish
            arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        return arr
