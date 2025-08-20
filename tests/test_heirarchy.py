def test_hierarchical_view(tmp_path):
    import numpy as np, os
    from zeromodel.pipeline.executor import PipelineExecutor
    rng = np.random.default_rng(0)
    N,M = 256, 64
    X = rng.normal(0,1,(N,M)).astype(np.float32)
    # Inject structure in a TL-ish block
    X[:64, :16] += 3.0 * rng.normal(0,1,(64,1)).astype(np.float32)

    out_final = str(tmp_path / "hier_final.png")
    out_prev  = str(tmp_path / "hier_preview.png")

    stages = [
      {"stage": "organizer/hierarchical_view.HierarchicalView", "params": {"levels": 3}},
      {"stage": "vpm/write.VPMWrite", "params": {"output_path": out_final}},
      {"stage": "vpm/write.VPMWrite", "params": {"output_path": out_prev, "array_key": "hierview.preview"}},
    ]
    Y, ctx = PipelineExecutor(stages).run(X, context={})
    print("Final shape:", Y.shape)
    assert os.path.exists(out_final)
    print(f"Final file written {out_final}")
    assert os.path.exists(out_prev)
    print(f"Preview file written {out_prev}")
