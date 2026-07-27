from pathlib import Path

from visual_transition_benchmark import baselines as bl
from visual_transition_benchmark import dataset as ds
from visual_transition_benchmark import report as rp
from visual_transition_benchmark import zeromodel_adapter as zm
from visual_transition_benchmark.render import build_html_index, render_transition_panel


def test_small_deterministic_dataset_end_to_end(tmp_path: Path):
    dev = ds.generate_split(prefix="smoke-dev", episode_count=2, seed_offset=0)
    eva = ds.generate_split(prefix="smoke-eval", episode_count=2, seed_offset=500)
    ds.assert_disjoint_splits(dev, eva)
    assert len(eva.records) == 2 * len(ds.ALL_CATEGORIES)
    assert any(r.is_faulty for r in eva.records)
    assert any(not r.is_faulty for r in eva.records)

    analyzer = zm.ArcadeBandZeroModelAnalyzer()
    zm_outputs, pd_outputs, priv_outputs = [], [], []
    for record in eva.records:
        metadata = zm.TransitionMetadata(
            transition_id=record.transition_id, step_number=record.step_number
        )
        zm_outputs.append(
            analyzer.analyze(
                record.frame_before, record.frame_after, record.action, metadata
            )
        )
        pd_outputs.append(
            bl.pixel_diff_baseline(record.frame_before, record.frame_after)
        )
        priv_outputs.append(bl.privileged_baseline(record))

    metrics_report = rp.build_metrics_report(
        eva.records, zm_outputs, pd_outputs, priv_outputs
    )
    assert metrics_report["all"]["n"] == len(eva.records)
    assert metrics_report["ordinary"]["n"] + metrics_report["faulty"]["n"] == len(
        eva.records
    )

    results_path = tmp_path / "benchmark-results.json"
    rp.write_json(results_path, {"metrics": metrics_report})
    assert results_path.exists()

    jsonl_path = tmp_path / "transition-level-results.jsonl"
    rp.write_transition_level_results(
        jsonl_path, eva.records, zm_outputs, pd_outputs, priv_outputs
    )
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(eva.records)

    faulty_record, faulty_zm, faulty_pd = next(
        (r, z, p) for r, z, p in zip(eva.records, zm_outputs, pd_outputs) if r.is_faulty
    )
    faulty_priv = priv_outputs[eva.records.index(faulty_record)]
    png_path = render_transition_panel(
        faulty_record,
        faulty_priv,
        faulty_zm,
        output_path=tmp_path / "artifacts" / "one.png",
    )
    assert png_path.exists() and png_path.stat().st_size > 0

    index_path = build_html_index(
        [
            {
                "transition_id": faulty_record.transition_id,
                "category": faulty_record.category,
                "fault_type": faulty_record.fault_type,
                "verdict": "better",
                "zeromodel_status": faulty_zm.diagnostics["conformance_status"],
                "artifact_path": "artifacts/one.png",
            }
        ],
        output_path=tmp_path / "visual-index.html",
    )
    assert index_path.exists()
    assert faulty_record.transition_id in index_path.read_text(encoding="utf-8")
