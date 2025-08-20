import os
import json
import numpy as np
from typing import Dict, Any, List
from zeromodel.pipeline.executor import PipelineStage

class FileReadFrames(PipelineStage):
    """
    Reads frames from {run_dir}/meta.jsonl where each line is:
      {"step": int, "path": ".../frames/000123.npy", "tags": {...}}
    Puts a list of dicts under context["frames_in"].
    """

    def __init__(self, run_dir: str, **kwargs):
        super().__init__(run_dir=run_dir, **kwargs)
        self.run_dir = run_dir

    def validate_params(self) -> None:
        if not isinstance(self.run_dir, str) or not self.run_dir:
            raise ValueError("run_dir must be a non-empty string")

    def process(self, X, context: Dict[str, Any]):
        meta_path = os.path.join(self.run_dir, "meta.jsonl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.jsonl not found at {meta_path}")
        out: List[Dict[str, Any]] = []
        with open(meta_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                fpath = rec["path"]
                frame = np.load(fpath).astype("float32")  # (H,W,C) in [0,1]
                out.append({"step": rec["step"], "frame": frame, "tags": rec.get("tags", {})})
        context["frames_in"] = out
        return X, context
