# zeromodel/pipeline/provenance/vpm_write_with_vpf.py
from __future__ import annotations
import os
import numpy as np
from typing import Any, Dict, Tuple
from PIL import Image

from zeromodel.pipeline.base import PipelineStage

# stripe + embed utilities from your provenance package
from zeromodel.provenance.stripe import add_visual_stripe  # visual right-edge stripe
from zeromodel.provenance.core import embed_vpf        # write iTXt 'vpf' (and footer if stripe)
from zeromodel.utils import tile_to_pil


class VPMWriteWithVPF(PipelineStage):
    """
    Serialize current VPM (numpy array) to PNG, embedding VPF and painting a right-edge stripe.
    Params:
      output_path: where to write the PNG
      add_stripe:  bool (default True)
      compress_itxt: bool (default False) – compress the iTXt 'vpf' payload
    """
    name = "vpm_write_with_vpf"
    category = "vpm"

    def __init__(self, **params):
        super().__init__(**params)
        self.output_path = params.get("output_path", "artifact.vpm.png")
        self.add_stripe = bool(params.get("add_stripe", True))
        self.compress_itxt = bool(params.get("compress_itxt", False))

    def validate_params(self):
        pass

    # replace _np_to_pil with a version that lets Pillow infer mode from array
    def _np_to_pil(self, arr: np.ndarray) -> Image.Image:
        # float arrays are assumed in [0,1]; scale to 8-bit
        def to_uint8(x):
            return (np.clip(x, 0, 1) * 255).astype(np.uint8)

        if arr.ndim == 2:
            x = to_uint8(arr) if arr.dtype.kind == "f" else arr
            return Image.fromarray(x)                     # let Pillow infer "L"
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            x = to_uint8(arr) if arr.dtype.kind == "f" else arr
            return Image.fromarray(x)                     # infer "RGB"/"RGBA"

        # fallback: visualize as single-channel
        x2d = arr if arr.ndim == 2 else arr.reshape(arr.shape[0], -1)
        x2d = to_uint8(x2d) if x2d.dtype.kind == "f" else x2d
        return Image.fromarray(x2d)

    def process(self, vpm: np.ndarray, context: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctx = self.get_context(context)
        vpf = ctx.get("vpf") or {}  # stage before should attach vpf dict

        base =  tile_to_pil(vpm)

        # Paint stripe visually if requested (safe even if vpf empty)
        img_for_embed = add_visual_stripe(base, vpf) if self.add_stripe else base

        # Embed the full VPF into iTXt 'vpf'; if we added stripe, footer mode is already handled in embed_vpf
        png_bytes = embed_vpf(
            img_for_embed,
            vpf,
            add_stripe=self.add_stripe,
            compress=self.compress_itxt,
            mode="stripe" if self.add_stripe else None,
        )

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "wb") as f:
            f.write(png_bytes)

        ctx["artifact_path"] = self.output_path
        meta = {
            "output_path": self.output_path,
            "add_stripe": self.add_stripe,
            "compress_itxt": self.compress_itxt,
            "bytes": len(png_bytes),
        }
        return vpm, meta
