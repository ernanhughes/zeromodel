from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple
from zeromodel.pipeline.executor import PipelineStage
from zeromodel.pipeline.organizer.top_left import TopLeft as TopLeftStage

def _top_left_mass(A: np.ndarray, alpha: float = 0.97) -> float:
    A = A.astype(np.float32, copy=False)
    r = alpha ** np.arange(A.shape[0], dtype=np.float32)
    c = alpha ** np.arange(A.shape[1], dtype=np.float32)
    W = np.outer(r, c)
    return float((A * W).sum() / (np.abs(A).sum() + 1e-8))

def _minmax_per_col(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lo = np.percentile(X, 1.0, axis=0, keepdims=True)
    hi = np.percentile(X, 99.0, axis=0, keepdims=True)
    hi = np.maximum(hi, lo + eps)
    return np.clip((X - lo) / (hi - lo), 0.0, 1.0)

class TLZoom(PipelineStage):
    """
    Hierarchical TopLeft zoom:
      Level 0: normalize(X) -> TopLeft -> pick best window (by TL-mass)
      Level 1..L-1: select rows/cols *in ORIGINAL ORDER* within that window,
                    re-normalize *on the original values of that window*,
                    run TopLeft again, pick next window ...
    Returns final organized submatrix + full hierarchy meta.

    Params:
      levels: int >=1
      row_frac, col_frac: crop fractions (0<..<=1)
      window_mode: 'quadrants' | 'top' | 'left' | 'halves'
        - 'quadrants': score TL-mass in [TL, TR, BL, BR] and pick max
        - 'top'/'left': pick top rows or left cols (use both if both chosen)
        - 'halves': compare (top half) vs (left half) vs (TL quadrant), pick max
      alpha: TL-mass decay
      top_left_params: dict forwarded to TopLeft at each level
      ref_feed: 'orig' | 'sorted'
        - 'orig'  : re-feed next level using ORIGINAL order subset (recommended)
        - 'sorted': re-feed next level using the *sorted* order from prev level

    Context adds:
      'tlzoom': {
         'levels': [... per-level dict ...],
         'final_rows_orig': [...], 'final_cols_orig': [...],
         'final_rows_sorted': [...], 'final_cols_sorted': [...]
      }
    """

    def __init__(self,
                 levels: int = 2,
                 row_frac: float = 0.5,
                 col_frac: float = 0.5,
                 window_mode: str = "quadrants",
                 alpha: float = 0.97,
                 ref_feed: str = "orig",
                 top_left_params: Dict[str, Any] | None = None,
                 **kw):
        super().__init__(levels=levels, row_frac=row_frac, col_frac=col_frac,
                         window_mode=window_mode, alpha=alpha, ref_feed=ref_feed,
                         top_left_params=top_left_params or {}, **kw)
        self.levels = int(levels)
        self.row_frac = float(row_frac)
        self.col_frac = float(col_frac)
        self.window_mode = window_mode
        self.alpha = float(alpha)
        self.ref_feed = ref_feed
        self.tl_params = dict(top_left_params or {})

    def validate_params(self):
        assert self.levels >= 1
        assert 0.0 < self.row_frac <= 1.0
        assert 0.0 < self.col_frac <= 1.0
        assert self.window_mode in ("quadrants", "top", "left", "halves")
        assert self.ref_feed in ("orig", "sorted")

    # --- helpers ---
    def _choose_window(self, Y: np.ndarray, rf: float, cf: float) -> Tuple[slice, slice, str, float]:
        H, W = Y.shape[:2]
        rh = max(1, int(round(H * rf)))
        cw = max(1, int(round(W * cf)))

        # Candidate windows in *sorted space*
        cands = []
        if self.window_mode == "quadrants":
            cands = [
                (slice(0, rh),        slice(0, cw),        "TL"),
                (slice(0, rh),        slice(W - cw, W),    "TR"),
                (slice(H - rh, H),    slice(0, cw),        "BL"),
                (slice(H - rh, H),    slice(W - cw, W),    "BR"),
            ]
        elif self.window_mode == "top":
            cands = [(slice(0, rh), slice(0, W), "TOP")]
        elif self.window_mode == "left":
            cands = [(slice(0, H), slice(0, cw), "LEFT")]
        elif self.window_mode == "halves":
            cands = [
                (slice(0, rh), slice(0, W), "TOP"),
                (slice(0, H),  slice(0, cw), "LEFT"),
                (slice(0, rh), slice(0, cw), "TL"),
            ]

        best_tag, best_s, best_val = None, None, -1.0
        for rs, cs, tag in cands:
            v = _top_left_mass(Y[rs, cs], self.alpha)
            if v > best_val:
                best_val = v
                best_s = (rs, cs)
                best_tag = tag
        return best_s[0], best_s[1], best_tag, best_val

    def process(self, X: np.ndarray, context: Dict[str, Any]):
        # Keep original matrix for each level’s re-normalization
        X0 = X.astype(np.float32, copy=False)
        N, M = X0.shape
        # At start, current candidate *orig* indices are full ranges
        rows_orig = np.arange(N, dtype=np.int32)
        cols_orig = np.arange(M, dtype=np.int32)

        hierarchy: List[Dict[str, Any]] = []
        feed = X0  # initial feed is original

        for lvl in range(self.levels):
            # (A) Re-normalize *on the original values* of the current subset
            #     regardless of previous TopLeft stretching.
            X_sub_orig = X0[np.ix_(rows_orig, cols_orig)]
            Xn = _minmax_per_col(X_sub_orig)

            # (B) Run TopLeft on the normalized subset
            tl = TopLeftStage(**self.tl_params)
            tl.validate_params()
            Y, meta = tl.process(Xn, context={})  # Y is sorted+pushed+stretched view

            # (C) Choose best window in *sorted space*
            rs, cs, tag, score = self._choose_window(Y, self.row_frac, self.col_frac)

            # (D) Map sorted-space window back to *original* indices
            # meta perms are new->orig, and Y is new-order:
            row_new_to_orig = np.array(meta["row_perm_new_to_orig"], dtype=np.int32)
            col_new_to_orig = np.array(meta["col_perm_new_to_orig"], dtype=np.int32)

            sel_rows_orig = row_new_to_orig[rs]
            sel_cols_orig = col_new_to_orig[cs]

            # For reference, also keep sorted indices (new order) that define the window
            sel_rows_sorted = np.arange(Y.shape[0], dtype=np.int32)[rs]
            sel_cols_sorted = np.arange(Y.shape[1], dtype=np.int32)[cs]

            # Record level info
            hierarchy.append({
                "level": lvl,
                "subset_shape_orig": tuple(X_sub_orig.shape),
                "tl_score": score,
                "window_tag": tag,
                "rows_orig": rows_orig.tolist(),
                "cols_orig": cols_orig.tolist(),
                "row_perm_new_to_orig": row_new_to_orig.tolist(),
                "col_perm_new_to_orig": col_new_to_orig.tolist(),
                "sel_rows_orig": sel_rows_orig.tolist(),
                "sel_cols_orig": sel_cols_orig.tolist(),
                "sel_rows_sorted": sel_rows_sorted.tolist(),
                "sel_cols_sorted": sel_cols_sorted.tolist(),
                "top_left_meta": meta,
            })

            # (E) Decide next-level feed indices
            rows_orig = sel_rows_orig
            cols_orig = sel_cols_orig

            # (F) If ref_feed=='sorted', convert selection into *sorted* feed
            if self.ref_feed == "sorted":
                # Build next-level matrix from Y subwindow (already normalized/pushed)
                # and treat those rows/cols as the new “orig” (we still keep the true
                # orig indices in hierarchy for traceability).
                feed = Y[rs, cs].astype(np.float32, copy=False)
            else:
                # Recommended: use original-order subset at next level
                feed = X0[np.ix_(rows_orig, cols_orig)]

            # End loop — next iteration will re-normalize `feed` properly from X0’s values.

        # Final organized submatrix (last selection), with a final TopLeft for presentation
        final_sub_orig = X0[np.ix_(rows_orig, cols_orig)]
        final_norm = _minmax_per_col(final_sub_orig)
        final_tl = TopLeftStage(**self.tl_params)
        final_tl.validate_params()
        final_Y, final_meta = final_tl.process(final_norm, context={})

        # Build context
        context.setdefault("tlzoom", {})
        context["tlzoom"]["levels"] = hierarchy
        context["tlzoom"]["final_rows_orig"] = rows_orig.tolist()
        context["tlzoom"]["final_cols_orig"] = cols_orig.tolist()
        context["tlzoom"]["final_rows_sorted"] = list(range(final_Y.shape[0]))
        context["tlzoom"]["final_cols_sorted"] = list(range(final_Y.shape[1]))
        context["tlzoom"]["final_top_left"] = final_meta

        return final_Y.astype(np.float32), context
