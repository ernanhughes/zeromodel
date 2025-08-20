# zeromodel/pipeline/combiner/masking.py
"""
StructuredMaskStage
-------------------
Pipeline stage for generating structured boolean masks over spatiotemporal 
patches with groups (T, Hp, Wp, G).

Implements structured masking in three domains:
- Time   (mask entire time slices)
- Group  (mask entire feature/metric groups)
- Space  (mask random spatial patches)

Guarantees an exact global mask ratio by adjusting the final mask.
Useful for: robustness tests, logical pruning, curriculum exposure,
and partial observability experiments.

Author: ZeroModel / Stephanie Integration
"""

import logging
from typing import Any, Dict, Tuple

import torch
from zeromodel.pipeline.executor import PipelineStage

logger = logging.getLogger(__name__)


class StructuredMaskStage(PipelineStage):
    """
    PipelineStage: Generate a structured mask tensor of shape (T, Hp, Wp, G).

    Parameters
    ----------
    mask_ratio : float
        Desired global fraction of masked elements [0.0, 1.0].
    time_frac : float
        Fraction of time bins to mask entirely (default=0.3).
    group_frac : float
        Fraction of groups to mask entirely (default=0.3).
    space_frac : float
        Probability of masking individual spatial patches (default=0.4).
    device : str or torch.device, optional
        Device on which to allocate mask tensor.
    """

    def __init__(
        self,
        mask_ratio: float = 0.75,
        time_frac: float = 0.3,
        group_frac: float = 0.3,
        space_frac: float = 0.4,
        device: str = None,
        **kwargs
    ):
        super().__init__(
            mask_ratio=mask_ratio,
            time_frac=time_frac,
            group_frac=group_frac,
            space_frac=space_frac,
            device=device,
            **kwargs,
        )
        self.mask_ratio = mask_ratio
        self.time_frac = time_frac
        self.group_frac = group_frac
        self.space_frac = space_frac
        self.device = device

    def validate_params(self) -> None:
        if not (0.0 <= self.mask_ratio <= 1.0):
            raise ValueError("mask_ratio must be between 0 and 1")
        for name, val in [
            ("time_frac", self.time_frac),
            ("group_frac", self.group_frac),
            ("space_frac", self.space_frac),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1")
        logger.debug(
            f"[StructuredMaskStage] Params validated: "
            f"mask_ratio={self.mask_ratio}, time_frac={self.time_frac}, "
            f"group_frac={self.group_frac}, space_frac={self.space_frac}"
        )

    def _structured_mask(self, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """Generate structured mask tensor."""
        T, Hp, Wp, G = shape
        m = torch.zeros(shape, dtype=torch.bool, device=self.device)

        # 1) Time masking
        t_hide = int(round(T * self.time_frac))
        if t_hide > 0:
            tidx = torch.randperm(T, device=self.device)[:t_hide]
            m[tidx, :, :, :] = True
            logger.debug(f"[StructuredMaskStage] Time-masked {t_hide}/{T} slices")

        # 2) Group masking
        g_hide = int(round(G * self.group_frac))
        if g_hide > 0:
            gidx = torch.randperm(G, device=self.device)[:g_hide]
            m[:, :, :, gidx] = True
            logger.debug(f"[StructuredMaskStage] Group-masked {g_hide}/{G} groups")

        # 3) Spatial masking
        if self.space_frac > 0:
            space = (torch.rand((T, Hp, Wp), device=self.device) < self.space_frac)
            m |= space[..., None]  # Broadcast over groups
            logger.debug(
                f"[StructuredMaskStage] Applied spatial masking with prob={self.space_frac}"
            )

        # 4) Adjust to exact global ratio
        total = T * Hp * Wp * G
        need = int(round(self.mask_ratio * total))
        flat = m.view(-1)
        cur = int(flat.sum().item())

        if cur < need:
            # randomly add masks
            unmasked_idx = (~flat).nonzero(as_tuple=False).view(-1)
            add = need - cur
            sel = unmasked_idx[torch.randperm(unmasked_idx.numel(), device=self.device)[:add]]
            flat[sel] = True
            logger.debug(f"[StructuredMaskStage] Added {add} masks to reach target ratio")
        elif cur > need:
            # randomly remove masks
            masked_idx = flat.nonzero(as_tuple=False).view(-1)
            rem = cur - need
            sel = masked_idx[torch.randperm(masked_idx.numel(), device=self.device)[:rem]]
            flat[sel] = False
            logger.debug(f"[StructuredMaskStage] Removed {rem} masks to reach target ratio")

        return m.view(T, Hp, Wp, G)

    def process(self, X: Any, context: Dict[str, Any]):
        """
        Process step: generates a structured mask given context["mask_shape"].

        Expects
        -------
        context["mask_shape"] : tuple
            Shape (T, Hp, Wp, G) describing mask dimensions.

        Writes
        ------
        context["mask"] : torch.Tensor
            Boolean mask of given shape.
        """
        shape = context.get("mask_shape")
        if shape is None:
            raise KeyError("context['mask_shape'] must be provided for StructuredMaskStage")

        mask = self._structured_mask(shape)
        context["mask"] = mask
        logger.info(f"[StructuredMaskStage] Mask generated with global ratio={self.mask_ratio:.2f}")
        return X, context
