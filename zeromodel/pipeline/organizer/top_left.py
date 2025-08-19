"""
TopLeft Organizer Module

Organizes image data by iteratively sorting rows and columns to push high-intensity
values toward the top-left corner (or specified corner) of the image.

Key Features:
- Multiple intensity aggregation methods (luminance, mean, max, min, etc.)
- Configurable number of sorting iterations
- Monotonic energy push toward specified corner
- Contrast stretching options
- Support for various image formats (2D, 3D, 4D)

Typical Use Cases:
- Preprocessing for visualization
- Emphasizing important regions in images
- Preparing data for downstream processing
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Iterable, Tuple
import numpy as np
from zeromodel.pipeline.base import PipelineStage

# Configure logger
logger = logging.getLogger(__name__)

# ---------- Intensity Aggregation Helpers ----------

def _aggregate_intensity(arr: np.ndarray,
                         metric_mode: str,
                         channel_weights: Iterable[float] | None) -> np.ndarray:
    """
    Reduce (H,W,C) → (H,W) per-pixel intensity according to the specified metric,
    or pass-through (H,W) unchanged.
    
    Args:
        arr: Input array with shape (H,W) or (H,W,C)
        metric_mode: Method for intensity calculation
        channel_weights: Optional weights for channel combination
        
    Returns:
        Intensity array with shape (H,W)
        
    Raises:
        ValueError: For invalid inputs or modes
    """
    logger.debug(f"Aggregating intensity with mode '{metric_mode}', shape: {arr.shape}")
    
    if arr.ndim == 2:
        logger.debug("Input is already 2D, returning as-is")
        return arr

    if arr.ndim != 3:
        error_msg = f"Expected (H,W) or (H,W,C), got {arr.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    H, W, C = arr.shape
    mode = (metric_mode or "luminance").lower()
    logger.debug(f"Processing {H}x{W}x{C} array with mode: {mode}")

    if mode == "luminance":
        # ITU-R BT.709 weights; requires >= 3 channels
        if C < 3:
            logger.warning(f"Luminance mode requires ≥3 channels, using mean instead (C={C})")
            return arr.mean(axis=-1)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        result = 0.2126 * r + 0.7152 * g + 0.0722 * b
        logger.debug("Applied luminance weights (BT.709)")
        return result

    if mode == "mean":
        result = arr.mean(axis=-1)
        logger.debug("Calculated channel mean")
        return result

    if mode == "sum":
        result = arr.sum(axis=-1)
        logger.debug("Calculated channel sum")
        return result

    if mode == "max":
        result = arr.max(axis=-1)
        logger.debug("Calculated channel max")
        return result

    if mode == "min":
        result = arr.min(axis=-1)
        logger.debug("Calculated channel min")
        return result

    if mode.startswith("channel:"):
        tag = mode.split(":", 1)[1].strip().lower()
        chmap = {"r": 0, "g": 1, "b": 2}
        k = chmap.get(tag, None)
        if k is None:
            try:
                k = int(tag)
            except ValueError:
                error_msg = f"Invalid channel specification: {tag}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        if not (0 <= k < C):
            error_msg = f"channel index {k} out of range for C={C}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"Selected channel {k} ({tag})")
        return arr[..., k]

    if mode == "weights":
        if channel_weights is None:
            error_msg = "metric_mode='weights' requires channel_weights"
            logger.error(error_msg)
            raise ValueError(error_msg)
        w = np.asarray(list(channel_weights), dtype=np.float32)
        if w.ndim != 1 or w.shape[0] != C:
            error_msg = f"channel_weights must have length {C}, got {w.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        w = np.maximum(0.0, w)
        s = float(w.sum()) or 1.0
        w = w / s
        logger.debug(f"Applied channel weights: {w}")
        return (arr * w[None, None, :]).sum(axis=-1)

    error_msg = f"Unknown metric_mode '{metric_mode}'"
    logger.error(error_msg)
    raise ValueError(error_msg)


# ---------- TopLeft Organizer Class ----------

class TopLeft(PipelineStage):
    """
    Alternating top-left organizer.

    Iteratively:
      1) Sort columns by column intensity (desc if reverse=True)
      2) Sort rows   by row   intensity (desc if reverse=True)

    Then (optional):
      - Monotone push: cumulative energy push toward specified corner
      - Contrast stretch: percentile-based normalization

    Parameters
    ----------
    metric_mode: 'luminance'|'mean'|'sum'|'max'|'min'|'channel:R'|'channel:<k>'|'weights'
        Method for aggregating multi-channel images to single intensity values
    channel_weights: list[float]
        Required if metric_mode='weights', specifies weights for each channel
    iterations: int, default=5
        Number of alternating sort operations to perform
    reverse: bool, default=True
        True → bright to top-left (descending), False → dark to top-left (ascending)
    monotone_push: bool, default=True
        Whether to apply cumulative energy push toward corner
    clip_percent: float, default=0.01
        Percentile for contrast stretching clipping, in range [0, 0.5)
    stretch: bool, default=True
        Whether to apply contrast stretching after monotone push
    push_corner: str, default='tl'
        Corner to push energy toward: 'tl' (top-left), 'tr' (top-right),
        'bl' (bottom-left), 'br' (bottom-right)
    """

    name = "top_left_aggressive"
    category = "organizer"

    def __init__(self, **params):
        """Initialize TopLeft organizer with configuration parameters."""
        super().__init__(**params)
        self.metric_mode = params.get("metric_mode", "luminance")
        self.channel_weights = params.get("channel_weights")
        self.iterations = int(params.get("iterations", 5))
        self.reverse = bool(params.get("reverse", True))  # bright → TL
        self.monotone_push = bool(params.get("monotone_push", True))
        self.clip_percent = float(params.get("clip_percent", 0.01))
        self.stretch = bool(params.get("stretch", True))
        self.push_corner = params.get("push_corner", "tl")
        
        logger.info(f"Initialized TopLeft organizer: {self.__repr__()}")

    def __repr__(self):
        """Return string representation of organizer configuration."""
        return (f"TopLeft(metric_mode={self.metric_mode}, iterations={self.iterations}, "
                f"reverse={self.reverse}, monotone_push={self.monotone_push}, "
                f"clip_percent={self.clip_percent}, stretch={self.stretch}, "
                f"push_corner={self.push_corner})")

    def validate_params(self):
        """Validate configuration parameters before processing."""
        logger.debug("Validating parameters")
        assert self.iterations >= 1, "iterations must be ≥ 1"
        assert 0.0 <= self.clip_percent < 0.5, "clip_percent must be in [0, 0.5)"
        logger.debug("Parameters validated successfully")

    def _once(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply top-left organization to a single image frame.
        
        Args:
            img: Input image with shape (H,W) or (H,W,C), float32
            
        Returns:
            Tuple of (processed_image, metadata_dict)
        """
        logger.debug(f"Processing single frame with shape {img.shape}")
        
        # Calculate intensity for sorting
        intensity = _aggregate_intensity(img, self.metric_mode, self.channel_weights).astype(np.float32)
        logger.debug(f"Intensity range: [{intensity.min():.3f}, {intensity.max():.3f}]")
        
        out = img.astype(np.float32, copy=True)

        # Apply alternating row/column sorts
        logger.debug(f"Starting {self.iterations} iterations of alternating sorts")
        for i in range(self.iterations):
            # Sort columns by column intensity
            col_score = intensity.sum(axis=0)
            col_order = np.argsort(-col_score) if self.reverse else np.argsort(col_score)
            out = out[:, col_order] if out.ndim == 2 else out[:, col_order, :]
            intensity = intensity[:, col_order]
            logger.debug(f"Iteration {i+1}: sorted {len(col_order)} columns")

            # Sort rows by row intensity
            row_score = intensity.sum(axis=1)
            row_order = np.argsort(-row_score) if self.reverse else np.argsort(row_score)
            out = out[row_order] if out.ndim == 2 else out[row_order, :, :]
            intensity = intensity[row_order]
            logger.debug(f"Iteration {i+1}: sorted {len(row_order)} rows")

        # Apply monotone push if enabled
        if self.monotone_push:
            logger.debug(f"Applying monotone push toward {self.push_corner} corner")
            if out.ndim == 2:
                # 2D case: apply cumulative sum directly
                Y = self._corner_cumsum(out, self.push_corner).astype(np.float32)
                m, M = float(Y.min()), float(Y.max())
                if M > m:
                    Y = (Y - m) / (M - m)
                    Y *= max(1e-12, float(out.max()))
                out = Y
                logger.debug("Applied monotone push to 2D array")
            else:
                # 3D case: use intensity to guide push
                I0 = _aggregate_intensity(out, self.metric_mode, self.channel_weights).astype(np.float32) + 1e-12
                I = self._corner_cumsum(I0, self.push_corner)
                I = I / (float(I.max()) + 1e-12)
                out = out * I[..., None]
                logger.debug("Applied intensity-guided monotone push to 3D array")
                
                # Apply contrast stretching if enabled
                if self.stretch and out.size:
                    logger.debug(f"Applying contrast stretch with clip_percent={self.clip_percent}")
                    lo = float(np.quantile(out, self.clip_percent))
                    hi = float(np.quantile(out, 1.0 - self.clip_percent))
                    if hi <= lo:
                        hi = lo + 1e-6
                        logger.warning("Contrast stretch range too small, adjusting")
                    out = np.clip((out - lo) / (hi - lo), 0.0, 1.0, out=out)
                    logger.debug(f"Contrast stretch range: [{lo:.3f}, {hi:.3f}]")

        # Prepare metadata
        meta = {
            "iterations": self.iterations,
            "reverse": self.reverse,
            "monotone_push": self.monotone_push,
            "clip_percent": self.clip_percent,
            "stretch": self.stretch,
            "metric_mode": self.metric_mode,
            "input_shape": tuple(img.shape),
            "reordering_applied": True,
            "output_shape": tuple(out.shape),
        }
        if self.metric_mode == "weights" and self.channel_weights is not None:
            meta["channel_weights"] = list(map(float, self.channel_weights))
            
        logger.debug(f"Processing complete. Output shape: {out.shape}")
        return out, meta

    def process(self, vpm: np.ndarray, context: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process input data according to the configured organization strategy.
        
        Args:
            vpm: Input data with supported shape:
                 - 2D: (H, W) single image
                 - 3D: (H, W, C) multi-channel image or (T, H, W) image sequence
                 - 4D: (T, H, W, C) sequence of multi-channel images
            context: Optional processing context dictionary
            
        Returns:
            Tuple of (processed_data, metadata_dict)
            
        Raises:
            ValueError: For unsupported input dimensions
        """
        logger.info(f"Processing data with shape {vpm.shape}")
        
        # Handle context/provenance (support both old/new base names)
        if hasattr(self, "get_context"):
            context = self.get_context(context)
        else:
            context = self._get_context(context)

        # Record processing provenance
        prov_payload = {
            "iterations": self.iterations,
            "reverse": self.reverse,
            "monotone_push": self.monotone_push,
            "clip_percent": self.clip_percent,
            "stretch": self.stretch,
            "metric_mode": self.metric_mode,
            "push_corner": self.push_corner,
            "channel_weights": None if self.channel_weights is None else list(self.channel_weights),
        }
        self.record_provenance(context, self.name, prov_payload)
        # Process based on input dimensions
        x = vpm
        if x.ndim == 2:  # (H,W)
            logger.debug("Processing 2D array")
            out, meta = self._once(x.astype(np.float32))
            return out.astype(np.float32), meta

        if x.ndim == 3:
            # If last dim looks like channels, treat as (H,W,C); else (T,H,W)
            if x.shape[-1] in (1, 3, 4):
                logger.debug("Processing 3D array as (H,W,C) image")
                out, meta = self._once(x.astype(np.float32))
                return out.astype(np.float32), meta
            
            # (T,H,W) grayscale series
            logger.debug("Processing 3D array as (T,H,W) sequence")
            T = x.shape[0]
            frames = []
            for t in range(T):
                logger.debug(f"Processing frame {t+1}/{T}")
                f, _ = self._once(x[t].astype(np.float32))
                frames.append(f.astype(np.float32))
            out = np.stack(frames, axis=0)
            meta = {
                "applied_per_frame": True,
                "iterations": self.iterations,
                "reverse": self.reverse,
                "monotone_push": self.monotone_push,
                "clip_percent": self.clip_percent,
                "stretch": self.stretch,
                "metric_mode": self.metric_mode,
                "input_shape": tuple(x.shape),
                "reordering_applied": True,
                "output_shape": tuple(out.shape),
            }
            logger.info(f"Processed {T} frames successfully")
            return out, meta

        if x.ndim == 4:  # (T,H,W,C)
            logger.debug("Processing 4D array as (T,H,W,C) sequence")
            T = x.shape[0]
            frames = []
            for t in range(T):
                logger.debug(f"Processing frame {t+1}/{T}")
                f, _ = self._once(x[t].astype(np.float32))
                frames.append(f.astype(np.float32))
            out = np.stack(frames, axis=0)
            meta = {
                "applied_per_frame": True,
                "iterations": self.iterations,
                "reverse": self.reverse,
                "monotone_push": self.monotone_push,
                "clip_percent": self.clip_percent,
                "stretch": self.stretch,
                "metric_mode": self.metric_mode,
                "input_shape": tuple(x.shape),
                "output_shape": tuple(out.shape),
            }
            logger.info(f"Processed {T} multi-channel frames successfully")
            return out, meta

        error_msg = f"Unsupported VPM rank {x.ndim}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _corner_cumsum(self, A: np.ndarray, corner: str) -> np.ndarray:
        """
        Cumulative sum that converges toward the specified corner.
        
        Args:
            A: Input 2D array
            corner: Target corner - 'tl', 'tr', 'bl', or 'br'
            
        Returns:
            Cumulative sum array with same shape as A
            
        Raises:
            ValueError: For invalid corner specification
        """
        logger.debug(f"Calculating cumulative sum toward {corner} corner")
        
        if corner == "tl":
            B = np.cumsum(np.cumsum(A[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]
        elif corner == "tr":
            B = np.cumsum(np.cumsum(A[::-1, :], axis=0), axis=1)[::-1, :]
        elif corner == "bl":
            B = np.cumsum(np.cumsum(A[:, ::-1], axis=1), axis=0)[:, ::-1]
        elif corner == "br":
            B = np.cumsum(np.cumsum(A, axis=0), axis=1)
        else:
            error_msg = f"push_corner must be one of ['tl','tr','bl','br'], got {corner!r}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.debug(f"Cumulative sum complete. Range: [{B.min():.3f}, {B.max():.3f}]")
        return B