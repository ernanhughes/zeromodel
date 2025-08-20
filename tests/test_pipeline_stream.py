import os
import json
import numpy as np
from zeromodel.pipeline.executor import PipelineExecutor

def _write_fake_run(run_dir, T=12, H=64, W=64, C=3):
    os.makedirs(os.path.join(run_dir,"frames"), exist_ok=True)
    meta = os.path.join(run_dir, "meta.jsonl")
    with open(meta, "w") as f:
        for t in range(T):
            frame = np.random.rand(H, W, C).astype("float32")
            p = os.path.join(run_dir, "frames", f"{t:06d}.npy")
            np.save(p, frame)
            f.write(json.dumps({"step": t, "path": p, "tags": {"loss": float(1.0/(t+1))}}) + "\n")
    with open(os.path.join(run_dir,"run.json"),"w") as f:
        json.dump({"status":"closed","fps":8}, f)

def test_stream_pipeline(tmp_path):
    run_dir = str("C:/Users/ernan/Project/zeromodel/notebooks/artifacts/zm_run")
    _write_fake_run(run_dir, T=16, H=64, W=64, C=3)

    out_gif = str(tmp_path / "stream_overlay.gif")

    stages = [
        {"stage": "data/file_read.FileReadFrames", "params": {"run_dir": run_dir}},
        {"stage": "maestro/group_norm.GroupNormalizeFrames",
         "params": {"groups": [[0,3]]}},  # one group (all channels) for test
        {"stage": "maestro/online_encode.Encode", "params": {"L": 8}},
        {"stage": "observer/overlay_residual.OverlayResidual", "params": {"alpha": 0.45}},
        {"stage": "data/gif_write.GIFWrite", "params": {"output_path": out_gif, "fps": 8}},
    ]

    # X is unused here; pass a dummy matrix to satisfy the executor's signature
    X = np.zeros((4,4), dtype=np.float32)
    result, ctx = PipelineExecutor(stages).run(X, context={})

    assert os.path.exists(out_gif), "GIF not created"
    assert os.path.getsize(out_gif) > 0, "GIF is empty"

