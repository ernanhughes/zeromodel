"""
Logic combiner for ZeroModel.

This implements ZeroModel's "symbolic logic in the data" principle:
Instead of running a neural model, we run fuzzy logic on structured images.

Supports fuzzy logic ops over channels of a VPM:
- and, or, xor, add, subtract, nand, nor, not

Convention:
- Input VPM is HxW (2D) or HxWxC (3D). For binary/multi-operand ops, we
  fold over the last 'channel' dimension. For NOT, we apply elementwise NOT.
- All math is continuous in [0,1] via zeromodel.vpm_logic helpers.
- Optionally, you can binarize the result with a threshold after the op.

If you need to combine *two separate* VPMs, pass them stacked as channels:
np.stack([vpm_a, vpm_b], axis=-1)  -> shape (H, W, 2)
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage
from zeromodel.vpm.logic import (normalize_vpm, vpm_add, vpm_and, vpm_nand,
                                 vpm_nor, vpm_not, vpm_or, vpm_subtract,
                                 vpm_xor)

logger = logging.getLogger(__name__)


_OPS: Dict[str, Callable[..., np.ndarray]] = {
    "and": vpm_and,
    "or": vpm_or,
    "xor": vpm_xor,
    "add": vpm_add,
    "subtract": vpm_subtract,
    "nand": vpm_nand,
    "nor": vpm_nor,
    "not": vpm_not,
}


class LogicCombiner(PipelineStage):
    """
    Generic fuzzy logic combiner over channels of a VPM.

    Params:
        op: one of {'and','or','xor','add','subtract','nand','nor','not'}
        binarize_after: Optional[float] threshold in (0,1]; if set, output is binarized.
        reduce_mode: 'fold' (pairwise left fold) or 'native' (fast path for and/or/add).
                     Default 'fold' to keep semantics uniform across ops.
    """
    name = "logic"
    category = "combiner"

    def __init__(self, **params):
        super().__init__(**params)
        self.op: str = params.get("op", "and").lower()
        self.binarize_after: Optional[float] = params.get("binarize_after", None)
        self.reduce_mode: str = params.get("reduce_mode", "fold")

    # If your PipelineExecutor calls `validate_params()` before `process`,
    # this will be invoked; otherwise it's harmless to keep.
    def validate_params(self):
        self._validate_params()

    def _validate_params(self):
        if self.op not in _OPS:
            raise ValueError(f"Unknown logic op '{self.op}'. "
                             f"Valid ops: {sorted(_OPS.keys())}")
        if self.binarize_after is not None:
            if not (0.0 < float(self.binarize_after) <= 1.0):
                raise ValueError("binarize_after must be in (0,1] if provided")
        if self.reduce_mode not in ("fold", "native"):
            raise ValueError("reduce_mode must be 'fold' or 'native'")

    def process(
        self,
        vpm: np.ndarray,
        context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        context = self._get_context(context)

        # Normalize to [0,1] float
        vpm_norm = normalize_vpm(vpm)

        # Handle NOT specially (unary)
        if self.op == "not":
            out = vpm_not(vpm_norm)  # works for 2D or 3D
            out = self._maybe_binarize(out)
            meta = self._make_meta(vpm, out, channels_combined=(vpm.shape[-1] if vpm.ndim == 3 else 1))
            meta["operation"] = "NOT"
            return out, meta

        # Binary/multi-operand ops: require channel dimension
        if vpm_norm.ndim < 3 or vpm_norm.shape[-1] < 2:
            # Not enough operands available in a single array
            msg = "VPM needs >=2 channels for multi-operand logic; got shape {}".format(vpm.shape)
            logger.warning(msg)
            return vpm, {"warning": msg, "operation": self.op.upper()}

        # Reduce across channels (last dimension)
        out = self._reduce_over_channels(vpm_norm, self.op, self.reduce_mode)

        out = self._maybe_binarize(out)
        meta = self._make_meta(vpm, out, channels_combined=vpm.shape[-1])
        meta["operation"] = self.op.upper()
        return out, meta

    def _reduce_over_channels(self, vpm_norm: np.ndarray, op: str, mode: str) -> np.ndarray:
        """
        Reduce HxWxC into HxW via the requested logic op.
        - 'fold': left-fold using the operator (works for all ops)
        - 'native': fast path for AND/OR/ADD (min/max/sum-clip)
        """
        H, W, C = vpm_norm.shape

        if mode == "native" and op in ("and", "or", "add"):
            if op == "and":
                # fuzzy AND: min across channels
                return np.min(vpm_norm, axis=-1)
            elif op == "or":
                # fuzzy OR: max across channels
                return np.max(vpm_norm, axis=-1)
            else:  # add
                # additive with clip
                return np.clip(np.sum(vpm_norm, axis=-1), 0.0, 1.0)

        # Generic fold (pairwise) using vpm_logic functions
        fn = _OPS[op]
        acc = vpm_norm[..., 0]
        for k in range(1, C):
            acc = fn(acc, vpm_norm[..., k])
        return acc

    def _maybe_binarize(self, v: np.ndarray) -> np.ndarray:
        if self.binarize_after is None:
            return v
        thr = float(self.binarize_after)
        return (v >= thr).astype(np.float32)

    def _make_meta(self, inp: np.ndarray, out: np.ndarray, channels_combined: int) -> Dict[str, Any]:
        return {
            "input_shape": tuple(inp.shape),
            "output_shape": tuple(out.shape),
            "channels_combined": int(channels_combined),
            "op": self.op,
            "binarized": self.binarize_after is not None,
            "reduce_mode": self.reduce_mode,
        }
