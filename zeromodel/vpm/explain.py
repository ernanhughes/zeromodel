# zeromodel/explain.py
import numpy as np

class OcclusionVPMInterpreter:
    """
    Gradient-free explainability for ZeroModel VPMs.

    This interpreter perturbs small spatial regions (occlusion) on the *encoded VPM image*
    and measures the change in a proxy score computed directly from the image.
    It does NOT call zeromodel.get_decision() for each occlusion (there is no hook to
    score arbitrary, patched VPMs inside the model). Instead, it uses an image-based
    proxy that can be either position-agnostic (uniform prior) or biased to top-left,
    mirroring the model’s positional bias if desired.
    """

    def __init__(
        self,
        patch_h: int = 8,
        patch_w: int = 8,
        stride: int = 4,
        baseline: str | np.ndarray = "zero",     # "zero" | "mean" | np.ndarray(H,W,3)
        prior: str = "top_left",                  # "top_left" | "uniform"
        score_mode: str = "intensity",           # currently supports "intensity"
        context_rows: int | None = None,         # optionally limit scoring to top rows
        context_cols: int | None = None,         # optionally limit scoring to left cols
        channel_agg: str = "mean"                # "mean" | "max"
    ):
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.stride = int(stride)
        self.baseline = baseline
        self.prior = prior
        self.score_mode = score_mode
        self.context_rows = context_rows
        self.context_cols = context_cols
        self.channel_agg = channel_agg

    # -------------------- Internals --------------------

    def _positional_weights(self, H: int, W: int) -> np.ndarray:
        """Create a positional weight map."""
        if self.prior == "uniform":
            return np.ones((H, W), dtype=np.float32)

        # top-left prior resembling get_decision() bias
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dist = np.sqrt(yy**2 + xx**2)
        w = np.maximum(0.0, 1.0 - 0.3 * dist)
        if w.max() > 0:
            w /= w.max()  # normalize to [0,1]
        else:
            w[:] = 1.0
        return w

    def _make_baseline(self, vpm_uint8: np.ndarray) -> np.ndarray:
        """Construct an occlusion baseline (uint8, same shape as VPM)."""
        if isinstance(self.baseline, np.ndarray):
            base = self.baseline.astype(np.uint8, copy=False)
            if base.shape != vpm_uint8.shape:
                raise ValueError(
                    f"Custom baseline shape {base.shape} does not match VPM {vpm_uint8.shape}"
                )
            return base

        if self.baseline == "mean":
            m = int(np.round(vpm_uint8.mean()))
            return np.full_like(vpm_uint8, m, dtype=np.uint8)

        # default: zero baseline
        return np.zeros_like(vpm_uint8, dtype=np.uint8)

    def _luminance(self, vpm01: np.ndarray) -> np.ndarray:
        """Aggregate channels to a single luminance map in [0,1]."""
        if self.channel_agg == "max":
            return vpm01.max(axis=2)
        # default: mean
        return vpm01.mean(axis=2)

    def _proxy_score(self, vpm01: np.ndarray, weights: np.ndarray) -> float:
        """
        Image-based proxy score. Currently only 'intensity' mode:
        - Compute luminance
        - Optionally crop to context
        - Weighted average by positional weights
        """
        if self.score_mode != "intensity":
            # Extend here in the future if you add more scoring schemes
            raise ValueError(f"Unsupported score_mode: {self.score_mode}")

        H, W, _ = vpm01.shape
        lum = self._luminance(vpm01)

        # Optionally focus on a subregion (e.g., top rows)
        r = min(self.context_rows or H, H)
        c = min(self.context_cols or W, W)

        if r < H or c < W:
            lum = lum[:r, :c]
            w = weights[:r, :c]
        else:
            w = weights

        denom = float(w.sum()) + 1e-12
        return float((lum * w).sum() / denom)

    def _ensure_float01(self, vpm: np.ndarray) -> np.ndarray:
        """Ensure VPM is float32 in [0,1]."""
        if np.issubdtype(vpm.dtype, np.floating):
            # assume already in [0,1]
            return vpm.astype(np.float32, copy=False)
        # assume uint8 0..255
        return (vpm.astype(np.float32) / 255.0)

    # -------------------- Public API --------------------

    def explain(self, zeromodel):
        """
        Compute an occlusion importance map aligned to the VPM pixel grid.

        Returns:
            importance: (H, W) float32 map in [0,1]
            meta: dict with base score and settings
        """
        if getattr(zeromodel, "sorted_matrix", None) is None:
            raise ValueError("ZeroModel not prepared/processed yet (sorted_matrix is None).")

        # Get VPM; try to keep float32 if encoder supports it, else normalize from uint8
        vpm = zeromodel.encode()  # H x W x 3
        vpm01 = self._ensure_float01(vpm)
        H, W, _ = vpm01.shape

        weights = self._positional_weights(H, W)
        base_score = self._proxy_score(vpm01, weights)

        # Build an occlusion baseline (in uint8 space), then convert to [0,1]
        base_img_uint8 = self._make_baseline(
            (np.clip(vpm01, 0.0, 1.0) * 255.0).astype(np.uint8)
        )
        base_img01 = base_img_uint8.astype(np.float32) / 255.0

        imp = np.zeros((H, W), dtype=np.float32)

        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                y2 = min(H, y + self.patch_h)
                x2 = min(W, x + self.patch_w)

                patched = vpm01.copy()
                patched[y:y2, x:x2, :] = base_img01[y:y2, x:x2, :]

                occ_score = self._proxy_score(patched, weights)
                drop = max(0.0, base_score - occ_score)

                # Fill the occluded region with the measured drop
                imp[y:y2, x:x2] += drop

        # Normalize importance to [0,1]
        mx = float(imp.max())
        if mx > 0:
            imp /= mx

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
