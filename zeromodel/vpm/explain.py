"""
Visual Policy Map (VPM) Explainability Module

Implements gradient-free interpretability methods for ZeroModel VPMs using occlusion techniques.
This approach perturbs spatial regions of the encoded VPM image to identify critical areas
that influence the model's decision-making process.

Key Concepts:
- Occlusion: Systematically blocking parts of the image to measure impact
- Proxy Score: Image-based approximation of decision importance
- Positional Bias: Modeling the model's focus on top-left regions
"""

import numpy as np


class OcclusionVPMInterpreter:
    """
    Gradient-free explainability for ZeroModel Visual Policy Maps (VPMs).
    
    This interpreter uses occlusion sensitivity analysis to identify important
    regions in the VPM image. By perturbing small spatial patches and measuring
    changes in a proxy score, it highlights areas that significantly influence
    the model's decision-making.
    
    Unlike gradient-based methods, this approach:
    1. Requires no access to model internals
    2. Works directly on the encoded VPM image
    3. Models the model's positional bias toward top-left regions
    
    Attributes:
        patch_h (int): Occlusion patch height (pixels)
        patch_w (int): Occlusion patch width (pixels)
        stride (int): Stride between occlusion patches
        baseline (str|ndarray): Occlusion replacement ("zero", "mean", or custom array)
        prior (str): Positional bias model ("top_left" or "uniform")
        score_mode (str): Proxy scoring method ("intensity" only currently)
        context_rows (int|None): Restrict scoring to top N rows
        context_cols (int|None): Restrict scoring to left M columns
        channel_agg (str): Channel aggregation method ("mean" or "max")
    """

    def __init__(
        self,
        patch_h: int = 8,
        patch_w: int = 8,
        stride: int = 4,
        baseline: str | np.ndarray = "zero",  # "zero" | "mean" | custom array
        prior: str = "top_left",               # "top_left" | "uniform"
        score_mode: str = "intensity",         # Currently supports "intensity"
        context_rows: int | None = None,       # Limit scoring to top rows
        context_cols: int | None = None,       # Limit scoring to left columns
        channel_agg: str = "mean"              # "mean" | "max"
    ):
        """
        Initialize occlusion interpreter with analysis parameters.
        
        Args:
            patch_h: Height of occlusion patches (pixels)
            patch_w: Width of occlusion patches (pixels)
            stride: Step size between patch centers
            baseline: Patch replacement strategy:
                "zero" - Replace with black
                "mean" - Replace with image mean
                ndarray - Custom replacement image
            prior: Positional bias model:
                "top_left" - Emphasize top-left regions (matches ZeroModel bias)
                "uniform" - No positional bias
            score_mode: Proxy scoring method (currently only "intensity")
            context_rows: Restrict scoring to top N rows (None = all rows)
            context_cols: Restrict scoring to left M columns (None = all columns)
            channel_agg: Channel aggregation for luminance:
                "mean" - Average across RGB channels
                "max" - Take maximum across channels
        """
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.stride = int(stride)
        self.baseline = baseline
        self.prior = prior
        self.score_mode = score_mode
        self.context_rows = context_rows
        self.context_cols = context_cols
        self.channel_agg = channel_agg

    # -------------------- Internal Helpers --------------------

    def _positional_weights(self, H: int, W: int) -> np.ndarray:
        """
        Create positional weight map modeling ZeroModel's spatial bias.
        
        The "top_left" prior approximates ZeroModel's tendency to focus on 
        top-left regions when making decisions, matching the behavior of 
        ZeroModel.get_decision().
        
        Args:
            H: Image height
            W: Image width
            
        Returns:
            Weight map of shape (H, W) with values in [0,1]
        """
        if self.prior == "uniform":
            return np.ones((H, W), dtype=np.float32)

        # Create radial gradient from top-left corner
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        
        # Euclidean distance from top-left (0,0)
        dist = np.sqrt(yy**2 + xx**2)
        
        # Apply inverse distance weighting
        # - Distant pixels get lower weights
        # - Nearby pixels get higher weights
        w = np.maximum(0.0, 1.0 - 0.3 * dist)
        
        # Normalize to [0,1]
        if w.max() > 0:
            w /= w.max()
        else:
            w[:] = 1.0  # Fallback for empty images
            
        return w

    def _make_baseline(self, vpm_uint8: np.ndarray) -> np.ndarray:
        """
        Create occlusion baseline image in uint8 format.
        
        Args:
            vpm_uint8: Original VPM image (uint8)
            
        Returns:
            Baseline image of same shape as input
            
        Raises:
            ValueError: On custom baseline shape mismatch
        """
        # Custom baseline array
        if isinstance(self.baseline, np.ndarray):
            base = self.baseline.astype(np.uint8, copy=False)
            if base.shape != vpm_uint8.shape:
                raise ValueError(
                    f"Baseline shape {base.shape} != VPM shape {vpm_uint8.shape}"
                )
            return base

        # Mean value baseline
        if self.baseline == "mean":
            m = int(np.round(vpm_uint8.mean()))
            return np.full_like(vpm_uint8, m, dtype=np.uint8)

        # Default: Zero (black) baseline
        return np.zeros_like(vpm_uint8, dtype=np.uint8)

    def _luminance(self, vpm01: np.ndarray) -> np.ndarray:
        """
        Convert RGB VPM to luminance map (single channel).
        
        Args:
            vpm01: Normalized VPM in [0,1] (float32)
            
        Returns:
            Luminance map of shape (H, W)
        """
        if self.channel_agg == "max":
            return vpm01.max(axis=2)  # Maximum across channels
        return vpm01.mean(axis=2)     # Mean across channels (default)

    def _proxy_score(self, vpm01: np.ndarray, weights: np.ndarray) -> float:
        """
        Compute proxy decision score from VPM image.
        
        The intensity-based score approximates ZeroModel's decision importance
        by combining luminance with positional weights.
        
        Args:
            vpm01: Normalized VPM in [0,1] (float32)
            weights: Positional weight map
            
        Returns:
            Scalar score representing decision importance
            
        Raises:
            ValueError: On unsupported score_mode
        """
        if self.score_mode != "intensity":
            raise ValueError(f"Unsupported score_mode: {self.score_mode}")

        H, W, _ = vpm01.shape
        lum = self._luminance(vpm01)

        # Apply context window if specified
        if self.context_rows or self.context_cols:
            r = min(self.context_rows or H, H)
            c = min(self.context_cols or W, W)
            lum = lum[:r, :c]
            w = weights[:r, :c]
        else:
            w = weights

        # Weighted average of luminance
        denom = float(w.sum()) + 1e-12
        return float((lum * w).sum() / denom)

    def _ensure_float01(self, vpm: np.ndarray) -> np.ndarray:
        """
        Convert VPM to float32 in [0,1] range.
        
        Handles both float and uint8 inputs.
        
        Args:
            vpm: Input VPM image
            
        Returns:
            Normalized float32 image in [0,1]
        """
        if np.issubdtype(vpm.dtype, np.floating):
            return vpm.astype(np.float32, copy=False)
        return (vpm.astype(np.float32) / 255.0)

    # -------------------- Public API --------------------

    def explain(self, zeromodel) -> tuple[np.ndarray, dict]:
        """
        Compute occlusion importance map for a ZeroModel VPM.
        
        Process:
        1. Extract and normalize VPM image
        2. Compute base score (no occlusion)
        3. For each patch position:
            a. Occlude patch region
            b. Compute perturbed score
            c. Record importance as score drop
        4. Normalize importance map
        
        Args:
            zeromodel: Prepared ZeroModel instance with sorted_matrix
            
        Returns:
            importance: (H, W) float32 importance map in [0,1]
            meta: Dictionary with analysis metadata
            
        Raises:
            ValueError: If model not prepared or unsupported inputs
        """
        # Validate model state
        if getattr(zeromodel, "sorted_matrix", None) is None:
            raise ValueError("ZeroModel not prepared (sorted_matrix missing).")

        # Get and normalize VPM image
        vpm = zeromodel.encode()  # H x W x 3
        vpm01 = self._ensure_float01(vpm)
        H, W, _ = vpm01.shape

        # Compute positional weights and base score
        weights = self._positional_weights(H, W)
        base_score = self._proxy_score(vpm01, weights)

        # Create baseline image for occlusion
        base_img_uint8 = self._make_baseline(
            (np.clip(vpm01, 0.0, 1.0) * 255.0).astype(np.uint8)
        )
        base_img01 = base_img_uint8.astype(np.float32) / 255.0

        # Initialize importance map
        imp = np.zeros((H, W), dtype=np.float32)

        # Slide occlusion window across image
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                # Define patch boundaries
                y2 = min(H, y + self.patch_h)
                x2 = min(W, x + self.patch_w)
                
                # Apply occlusion
                patched = vpm01.copy()
                patched[y:y2, x:x2, :] = base_img01[y:y2, x:x2, :]
                
                # Compute score with occlusion
                occ_score = self._proxy_score(patched, weights)
                
                # Importance = base_score - occ_score (larger drop = more important)
                drop = max(0.0, base_score - occ_score)
                
                # Accumulate importance in occluded region
                imp[y:y2, x:x2] += drop

        # Normalize importance to [0,1]
        imp_max = imp.max()
        if imp_max > 0:
            imp /= imp_max

        # Analysis metadata
        meta = {
            "base_score": base_score,
            "prior": self.prior,
            "score_mode": self.score_mode,
            "patch_h": self.patch_h,
            "patch_w": self.patch_w,
            "stride": self.stride,
            "context_rows": self.context_rows,
            "context_cols": self.context_cols,
            "channel_agg": self.channel_agg,
        }
        return imp.astype(np.float32), meta