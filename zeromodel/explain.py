# zeromodel/explain.py
import numpy as np

class OcclusionVPMInterpreter:
    """
    Gradient-free explainability for ZeroModel decisions.
    It perturbs small spatial regions (occlusion) and measures
    change in relevance returned by zeromodel.get_decision().
    Higher drop => region is more important.
    """

    def __init__(self, patch_h=8, patch_w=8, stride=4, baseline="zero", prior="top_left"):
        """
        Args:
            patch_h, patch_w: occlusion patch size in pixels of the VPM image (H x W x 3)
            stride: sliding step in pixels
            baseline: "zeros" | "mean"  (how to fill the occluded patch)
        """
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.stride = stride
        self.baseline = baseline  # "zero" | "mean" | np.ndarray
        self.prior = prior        # "top_left" | "uniform"

    def _positional_weights(self, H, W):
        if self.prior == "uniform":
            return np.ones((H, W), dtype=np.float32)
        # top-left prior (same as earlier, but confined to pixel grid)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dist = np.sqrt(yy**2 + xx**2)
        w = np.maximum(0.0, 1.0 - 0.3 * dist)
        # avoid all-zero
        if w.max() > 0:
            w /= w.max()
        return w

    def _make_baseline(self, vpm_uint8):
        if isinstance(self.baseline, np.ndarray):
            return self.baseline.astype(np.uint8)
        if self.baseline == "mean":
            m = int(np.round(vpm_uint8.mean()))
            return np.full_like(vpm_uint8, m, dtype=np.uint8)
        # default: zero baseline
        return np.zeros_like(vpm_uint8, dtype=np.uint8)

    def explain(self, zeromodel):
        """
        Returns:
            importance: H x W float map aligned to VPM pixels (channel-aggregated)
            meta: dict with base_relevance and settings
        """
        if zeromodel.sorted_matrix is None:
            raise ValueError("ZeroModel not prepared/processed yet.")

        vpm = zeromodel.encode().astype(np.float32) / 255.0  # H x W x 3
        H, W, _ = vpm.shape

        weights = self._positional_weights(H, W)

        def proxy_score(img01):
            # average channels, apply chosen weights
            lum = img01.mean(axis=2)
            return float((lum * weights).sum() / (weights.sum() + 1e-12))

        base_proxy = proxy_score(vpm)

        base_img = self._make_baseline((vpm * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
        imp = np.zeros((H, W), dtype=np.float32)

        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                y2 = min(H, y + self.patch_h)
                x2 = min(W, x + self.patch_w)

                patched = vpm.copy()
                patched[y:y2, x:x2, :] = base_img[y:y2, x:x2, :]

                occ_proxy = proxy_score(patched)
                drop = max(0.0, base_proxy - occ_proxy)
                imp[y:y2, x:x2] += drop

        if imp.max() > 0:
            imp /= imp.max()
        return imp, {"base_proxy": base_proxy, "prior": self.prior,
                     "patch_h": self.patch_h, "patch_w": self.patch_w, "stride": self.stride}


    def build_synthetic_matrix(H=20, K=9):
        M = 3 * K
        mat = np.random.rand(H, M) * 0.1
        # inject strong signal in top-left window: first 5 docs, first K pixels (i.e., 3*K metrics)
        mat[:5, :3*K] += 0.8
        mat = np.clip(mat, 0, 1)
        names = [f"m{i}" for i in range(M)]
        return mat, names

    @staticmethod
    def _positional_weights(H, W):
        """
        Approximate the get_decision top-left weighting at pixel resolution.
        """
        weights = np.zeros((H, W), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                # distance in 'pixel space' — same spirit as doc/metric distance
                d = np.sqrt((i**2) + (j**2))
                w = max(0.0, 1.0 - 0.3*d)
                weights[i, j] = w
        # Avoid all-zero
        if weights.sum() == 0:
            weights[:] = 1.0
        return weights
