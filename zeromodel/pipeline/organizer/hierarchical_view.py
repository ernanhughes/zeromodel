from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

from zeromodel.pipeline.executor import PipelineStage
from zeromodel.pipeline.organizer.top_left import TopLeft as TopLeftStage

logger = logging.getLogger(__name__)


# ---------- helpers ----------

def _top_left_mass(A: np.ndarray, alpha: float = 0.97) -> float:
    """Weighted mass of matrix with exponential decay toward bottom-right."""
    A = A.astype(np.float32, copy=False)
    r = alpha ** np.arange(A.shape[0], dtype=np.float32)
    c = alpha ** np.arange(A.shape[1], dtype=np.float32)
    W = np.outer(r, c)
    denom = float(np.abs(A).sum()) + 1e-8
    return float((A * W).sum() / denom)


def _minmax_per_col(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Column-wise min-max normalization using robust 1–99% bounds."""
    lo = np.percentile(X, 1.0, axis=0, keepdims=True)
    hi = np.percentile(X, 99.0, axis=0, keepdims=True)
    hi = np.maximum(hi, lo + eps)
    return np.clip((X - lo) / (hi - lo), 0.0, 1.0)


def _normalize_subset(X_sub: np.ndarray, X_full: np.ndarray, mode: str) -> np.ndarray:
    """Normalize a submatrix according to different strategies."""
    if mode == "local_subset":
        return _minmax_per_col(X_sub)

    lo_f = float(np.percentile(X_full, 1.0))
    hi_f = float(np.percentile(X_full, 99.0))
    hi_f = max(hi_f, lo_f + 1e-6)

    if mode == "global_full":
        return np.clip((X_sub - lo_f) / (hi_f - lo_f), 0.0, 1.0)

    # hybrid_ref: shrink global window to subset robust bounds
    lo_s = float(np.percentile(X_sub, 5.0))
    hi_s = float(np.percentile(X_sub, 95.0))
    lo_eff = max(lo_f, lo_s)
    hi_eff = min(hi_f, hi_s)
    hi_eff = max(hi_eff, lo_eff + 1e-6)
    return np.clip((X_sub - lo_eff) / (hi_eff - lo_eff), 0.0, 1.0)


def _tile_h(panels: List[np.ndarray], pad: int = 2) -> np.ndarray:
    """Tile 2D panels horizontally with padding, each normalized for preview."""
    if not panels:
        return np.zeros((8, 8), dtype=np.float32)

    H = max(p.shape[0] for p in panels)
    outs = []
    for i, p in enumerate(panels):
        q = p.astype(np.float32, copy=False)
        if q.size:
            lo, hi = np.quantile(q, 0.01), np.quantile(q, 0.99)
            if hi <= lo:
                hi = lo + 1e-6
            q = np.clip((q - lo) / (hi - lo), 0.0, 1.0)
        # pad vertically if needed
        if q.shape[0] != H:
            pad_h = H - q.shape[0]
            q = np.pad(q, ((0, pad_h), (0, 0)), mode="edge")
        outs.append(q)

    # Insert padding columns
    pad_arrs = [np.zeros((H, pad), dtype=np.float32)] * (len(outs) - 1)
    tiled: List[np.ndarray] = []
    for i, p in enumerate(outs):
        tiled.append(p)
        if i < len(outs) - 1:
            tiled.append(pad_arrs[i])
    return np.concatenate(tiled, axis=1)


# ---------- core class ----------

@dataclass
class _Node:
    level: int
    rows_orig: np.ndarray
    cols_orig: np.ndarray
    tl_score: float
    tag: str
    panel_sorted: np.ndarray
    row_perm_new_to_orig: np.ndarray
    col_perm_new_to_orig: np.ndarray
    rs: slice
    cs: slice


class HierarchicalView(PipelineStage):
    """
    Build a hierarchical zoom view using TopLeft at each level.
    """

    def __init__(self,
                 levels: int = 3,
                 row_frac: float = 0.5,
                 col_frac: float = 0.5,
                 window_mode: str = "quadrants",
                 norm_mode: str = "local_subset",
                 alpha: float = 0.97,
                 beam_size: int = 1,
                 early_stop_epsilon: float = 1e-3,
                 top_left_params: Dict[str, Any] | None = None,
                 **kw):
        super().__init__(**kw)
        self.levels = int(levels)
        self.row_frac = float(row_frac)
        self.col_frac = float(col_frac)
        self.window_mode = window_mode
        self.norm_mode = norm_mode
        self.alpha = float(alpha)
        self.beam_size = int(beam_size)
        self.early_eps = float(early_stop_epsilon)
        self.tl_params = dict(top_left_params or {})

    def validate_params(self):
        assert self.levels >= 1
        assert 0.0 < self.row_frac <= 1.0
        assert 0.0 < self.col_frac <= 1.0
        assert self.window_mode in ("quadrants", "top", "left", "halves")
        assert self.norm_mode in ("local_subset", "global_full", "hybrid_ref")
        assert self.beam_size >= 1

    def _candidates(self, Y: np.ndarray) -> List[Tuple[slice, slice, str, float]]:
        """Enumerate candidate windows in sorted space."""
        H, W = Y.shape[:2]
        rh = max(1, int(round(H * self.row_frac)))
        cw = max(1, int(round(W * self.col_frac)))

        C: List[Tuple[slice, slice, str, float]] = []
        if self.window_mode == "quadrants":
            windows = [
                (slice(0, rh), slice(0, cw), "TL"),
                (slice(0, rh), slice(W-cw, W), "TR"),
                (slice(H-rh, H), slice(0, cw), "BL"),
                (slice(H-rh, H), slice(W-cw, W), "BR"),
            ]
        elif self.window_mode == "top":
            windows = [(slice(0, rh), slice(0, W), "TOP")]
        elif self.window_mode == "left":
            windows = [(slice(0, H), slice(0, cw), "LEFT")]
        else:  # halves
            windows = [
                (slice(0, rh), slice(0, W), "TOP"),
                (slice(0, H), slice(0, cw), "LEFT"),
                (slice(0, rh), slice(0, cw), "TL"),
            ]

        for rs, cs, tag in windows:
            score = _top_left_mass(Y[rs, cs], self.alpha)
            logger.debug(f"[Candidates] Window {tag}: score={score:.4f}, shape={Y[rs, cs].shape}")
            C.append((rs, cs, tag, score))

        C.sort(key=lambda t: t[3], reverse=True)
        return C

    def _run_level(self, X_full: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> List[_Node]:
        """Run TopLeft and candidate selection for one level."""
        logger.debug(f"[_run_level] Running on submatrix rows={rows.size}, cols={cols.size}")
        X_sub_orig = X_full[np.ix_(rows, cols)]

        # normalization
        Xn = _normalize_subset(X_sub_orig, X_full, self.norm_mode)
        logger.debug(f"[_run_level] Normalized subset, mode={self.norm_mode}, shape={Xn.shape}")

        # apply TopLeft
        tl = TopLeftStage(**self.tl_params)
        tl.validate_params()
        Y, meta = tl.process(Xn, context={})
        logger.debug(f"[_run_level] TopLeft applied, output shape={Y.shape}")

        # enumerate candidate windows
        cands = self._candidates(Y)

        # map indices back to original
        row_n2o = np.asarray(meta["row_perm_new_to_orig"], dtype=np.int32)
        col_n2o = np.asarray(meta["col_perm_new_to_orig"], dtype=np.int32)

        outs: List[_Node] = []
        for rs, cs, tag, score in cands[: self.beam_size]:
            sel_rows_orig = row_n2o[rs]
            sel_cols_orig = col_n2o[cs]
            node = _Node(
                level=0,
                rows_orig=sel_rows_orig.copy(),
                cols_orig=sel_cols_orig.copy(),
                tl_score=float(score),
                tag=tag,
                panel_sorted=Y,
                row_perm_new_to_orig=row_n2o.copy(),
                col_perm_new_to_orig=col_n2o.copy(),
                rs=rs, cs=cs,
            )
            logger.debug(f"[_run_level] Candidate {tag}, score={score:.4f}, rows={len(sel_rows_orig)}, cols={len(sel_cols_orig)}")
            outs.append(node)

        return outs

    def process(self, X: np.ndarray, context: Dict[str, Any]):
        logger.info(f"[HierarchicalView] Starting process, shape={X.shape}")
        X0 = X.astype(np.float32, copy=False)
        N, M = X0.shape

        init_rows = np.arange(N, dtype=np.int32)
        init_cols = np.arange(M, dtype=np.int32)

        # first level
        beam: List[_Node] = self._run_level(X0, init_rows, init_cols)
        history_best: List[Dict[str, Any]] = []
        preview_panels: List[np.ndarray] = []

        head = beam[0]
        history_best.append(self._node_meta(head, level=0))
        preview_panels.append(self._panel_crop(head))

        for lvl in range(1, self.levels):
            prev_tl = history_best[-1]["tl_score"]
            cur_tl = beam[0].tl_score
            if (cur_tl - prev_tl) < self.early_eps:
                logger.info(f"[process] Early stop at level {lvl}, improvement={cur_tl - prev_tl:.6f}")
                break

            # expand beam
            new_beam: List[_Node] = []
            for node in beam:
                kids = self._run_level(X0, node.rows_orig, node.cols_orig)
                for k in kids:
                    k.level = lvl
                new_beam.extend(kids)

            if not new_beam:
                logger.warning(f"[process] No candidates at level {lvl}, stopping.")
                break

            new_beam.sort(key=lambda n: n.tl_score, reverse=True)
            beam = new_beam[: self.beam_size]
            head = beam[0]

            logger.debug(f"[process] Level {lvl}, best tag={head.tag}, score={head.tl_score:.4f}")
            history_best.append(self._node_meta(head, level=lvl))
            preview_panels.append(self._panel_crop(head))

        # final focus
        final_rows = head.rows_orig
        final_cols = head.cols_orig
        final_sub = X0[np.ix_(final_rows, final_cols)]

        preview = _tile_h(preview_panels, pad=2)

        context.setdefault("hierview", {})
        context["hierview"]["levels"] = history_best
        context["hierview"]["preview"] = preview
        context["hierview"]["final_rows_orig"] = final_rows.tolist()
        context["hierview"]["final_cols_orig"] = final_cols.tolist()
        context["hierview"]["beam_size"] = self.beam_size

        logger.info(f"[HierarchicalView] Finished, final submatrix shape={final_sub.shape}")
        return final_sub.astype(np.float32), context

    def _panel_crop(self, node: _Node) -> np.ndarray:
        """Crop the chosen candidate panel for preview."""
        return node.panel_sorted[node.rs, node.cs].astype(np.float32, copy=False)

    def _node_meta(self, node: _Node, level: int) -> Dict[str, Any]:
        """Summarize metadata for a node."""
        return {
            "level": int(level),
            "tag": node.tag,
            "tl_score": float(node.tl_score),
            "rows_orig_len": int(node.rows_orig.size),
            "cols_orig_len": int(node.cols_orig.size),
            "row_perm_new_to_orig": node.row_perm_new_to_orig.tolist(),
            "col_perm_new_to_orig": node.col_perm_new_to_orig.tolist(),
            "rs": [node.rs.start or 0, node.rs.stop or 0],
            "cs": [node.cs.start or 0, node.cs.stop or 0],
        }
