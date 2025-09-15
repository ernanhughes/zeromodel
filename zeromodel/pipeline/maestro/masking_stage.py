#  zeromodel/pipeline/maestro/masking_stage.py
"""
MAESTRO-ZM Structured Masking Stage for ZeroModel pipelines.

This stage applies structured masking across time, space, and groups,
returning a boolean mask tensor with an exact global mask ratio.

Author: ZeroModel Project
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch
from zeromodel.pipeline.executor import PipelineStage

logger = logging.getLogger(__name__)


class MaskingStage(PipelineStage):
    """
    Structured masking pipeline stage.

    Args:
        mask_ratio (float): global fraction of masked elements (0–1)
        time_frac (float): fraction of time bins to mask
        group_frac (float): fraction of groups to mask
        space_frac (float): probability of masking each spatial patch
        device (str or torch.device): device for mask computation

    Input Context:
        context["frames_maestro"]: list of {"residual": np.ndarray (H, W)}
        context["mask_shape"]: optional tuple (T, Hp, Wp, G)

    Output Context:
        context["mask"]: torch.BoolTensor of shape (T, Hp, Wp, G)
    """

    name = "masking"

    def __init__(
        self,
        mask_ratio: float = 0.75,
        time_frac: float = 0.3,
        group_frac: float = 0.3,
        space_frac: float = 0.4,
        device: str = None,
        **kwargs,
    ):
        super().__init__(
            mask_ratio=mask_ratio,
            time_frac=time_frac,
            group_frac=group_frac,
            space_frac=space_frac,
            device=device,
            **kwargs,
        )
        self.mask_ratio = float(mask_ratio)
        self.time_frac = float(time_frac)
        self.group_frac = float(group_frac)
        self.space_frac = float(space_frac)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def validate_params(self) -> None:
        if not (0 <= self.mask_ratio <= 1):
            raise ValueError("mask_ratio must be in [0,1]")
        for k, v in [
            ("time_frac", self.time_frac),
            ("group_frac", self.group_frac),
            ("space_frac", self.space_frac),
        ]:
            if not (0 <= v <= 1):
                raise ValueError(f"{k} must be in [0,1]")

    def _structured_mask(self, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """Core masking logic."""
        T, Hp, Wp, G = shape
        m = torch.zeros(shape, dtype=torch.bool, device=self.device)

        # 1) Time masking
        t_hide = int(round(T * self.time_frac))
        if t_hide > 0:
            tidx = torch.randperm(T, device=self.device)[:t_hide]
            m[tidx, :, :, :] = True

        # 2) Group masking
        g_hide = int(round(G * self.group_frac))
        if g_hide > 0:
            gidx = torch.randperm(G, device=self.device)[:g_hide]
            m[:, :, :, gidx] = True

        # 3) Spatial masking
        if self.space_frac > 0:
            space = (
                torch.rand((T, Hp, Wp), device=self.device) < self.space_frac
            )
            m |= space[..., None]  # broadcast over groups

        # 4) Adjust to exact global ratio
        total = T * Hp * Wp * G
        need = int(round(self.mask_ratio * total))
        flat = m.view(-1)
        cur = int(flat.sum().item())

        if cur < need:
            # add extra masks
            unmasked_idx = (~flat).nonzero(as_tuple=False).view(-1)
            if unmasked_idx.numel() > 0:
                sel = unmasked_idx[
                    torch.randperm(unmasked_idx.numel(), device=self.device)[: (need - cur)]
                ]
                flat[sel] = True
        elif cur > need:
            # remove masks
            masked_idx = flat.nonzero(as_tuple=False).view(-1)
            if masked_idx.numel() > 0:
                sel = masked_idx[
                    torch.randperm(masked_idx.numel(), device=self.device)[: (cur - need)]
                ]
                flat[sel] = False

        return m.view(T, Hp, Wp, G)

    def process(self, X, context: Dict[str, Any]):
        frames = context.get("frames_maestro", [])
        if not frames:
            raise ValueError("[masking] No frames_maestro found in context")

        # Derive shape (T,H,W,G)
        T = len(frames)
        H, W = frames[0]["residual"].shape
        G = 1  # group dim, extend if needed

        # Option A: trust context if provided
        shape = context.get("mask_shape", (T, H, W, G))
        context["mask_shape"] = shape

        logger.debug("[masking] Using mask_shape=%s", shape)

        mask = self._structured_mask(shape)
        context["mask"] = mask

        logger.debug(
            "[%s] Generated mask with shape=%s, target=%.3f, actual=%.3f",
            self.name,
            shape,
            self.mask_ratio,
            float(mask.float().mean().item()),
        )

        return X, context
