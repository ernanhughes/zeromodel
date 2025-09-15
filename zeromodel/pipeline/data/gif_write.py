#  zeromodel/pipeline/data/gif_write.py
from __future__ import annotations
import os
from typing import Dict, Any
from PIL import Image, ImageDraw

class GIFWrite:
    """
    Writes a GIF from context['frames_overlay'] images.
    """
    name = "gif/write.GIFWrite"

    def __init__(self, output_path: str, fps: int = 8, annotate: bool = True):
        self.output_path = output_path
        self.duration = int(1000 / int(fps))
        self.annotate = bool(annotate)

    def validate_params(self):
        if not self.output_path:
            raise ValueError("Output path must be specified.")
        if self.duration <= 0:
            raise ValueError("Duration must be positive.")
        if not isinstance(self.annotate, bool):
            raise ValueError("Annotate must be a boolean.")

    def process(self, X, context: Dict[str, Any]):
        items = context.get("frames_overlay", [])
        if not items:
            context["gif_saved"] = None
            return X, context
        frames = []
        for it in items:
            im = it["image"].convert("P", palette=Image.ADAPTIVE, colors=256)
            if self.annotate:
                draw = ImageDraw.Draw(im)
                step = it.get("step", None)
                phase = it.get("phase", None)
                loss  = (it.get("tags") or {}).get("loss", None)
                text = f"step {step}"
                if phase is not None: text += f" | phase {phase}"
                if loss  is not None: text += f" | loss {loss:.3f}"
                draw.text((4,4), text, fill=255)
            frames.append(im)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        frames[0].save(self.output_path, save_all=True, append_images=frames[1:],
                       optimize=True, duration=self.duration, loop=0)
        context["gif_saved"] = self.output_path
        return X, context
