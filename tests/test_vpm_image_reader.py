# tests/test_vpm_image_tojson_roundtrip.py
import json
import pathlib

import numpy as np

from zeromodel.vpm.image import VPMImageReader, VPMImageWriter
from zeromodel.vpm.metadata import AggId, VPMMetadata


def _make_writer_png(tmp_path, *, M=3, D=16, store_minmax=True,
                     meta_doc_count=None, level=0, agg_id=AggId.RAW, fname="vpm.png"):
    """
    Create a VPM-IMG on disk using VPMImageWriter and return the path plus
    the exact score_matrix used.
    """
    # Use already-normalized rows so percentile math is deterministic
    rng = np.random.default_rng(1234)
    score_matrix = rng.random((M, D), dtype=np.float64)

    # Build a proper VMETA payload (so Reader.D_logical is meaningful)
    md = VPMMetadata.for_tile(
        level=level,
        metric_count=M,
        doc_count=int(meta_doc_count if meta_doc_count is not None else D),
        doc_block_size=1,
        agg_id=int(agg_id),
        metric_weights={},  # none
        metric_names=[f"m{i}" for i in range(M)],
        task_hash=0,
        tile_id=VPMMetadata.make_tile_id(b"unit-test"),
        parent_id=b"\x00" * 16,
    )
    meta_bytes = md.to_bytes()

    p = pathlib.Path(tmp_path) / fname
    writer = VPMImageWriter(
        score_matrix=score_matrix,         # shape (M, D)
        store_minmax=store_minmax,
        compression=3,
        level=level,
        doc_block_size=1,
        agg_id=int(agg_id),
        metadata_bytes=meta_bytes,
        metric_names=[f"m{i}" for i in range(M)],
    )
    writer.write(str(p))
    return str(p), score_matrix


def test_tojson_roundtrip_deterministic(tmp_path):
    """
    Write -> Read -> to_json -> save -> reload -> equal.
    Also sanity-check a few header fields.
    """
    path, _ = _make_writer_png(tmp_path, M=4, D=16, store_minmax=True, fname="deterministic.png")

    r1 = VPMImageReader(path)
    report1 = r1.to_json(include_data=True,
                         data_mode="raw_u16",
                         include_channels="RGB",
                         max_docs=None,
                         downsample=None,
                         pretty=False)

    json_path = pathlib.Path(tmp_path) / "dump.json"
    VPMImageReader.save_json(report1, str(json_path), pretty=False)

    # Reload saved JSON and compare
    with open(json_path, "r", encoding="utf-8") as f:
        report2 = json.load(f)

    assert report1 == report2, "Saved JSON must reload byte-for-byte equal"

    # Basic header sanity
    assert report1["format"] == "VPM-IMG/v1"
    assert report1["M"] == 4
    assert report1["D_physical"] >= 12  # writer guarantees header width >= META_MIN_COLS
    assert report1["D_logical"] == report1["vmeta"].get("doc_count", report1["D_physical"])


def test_tojson_matches_pixels_and_respects_logical_width(tmp_path):
    """
    Ensure to_json trims to D_logical and that channel arrays match
    the reader’s raw u16 pixels exactly on those columns.
    """
    D_phys = 16
    D_logical = 10  # embed logical width smaller than physical (padding case)

    path, _ = _make_writer_png(
        tmp_path,
        M=3,
        D=D_phys,
        store_minmax=False,                 # simpler: no denorm path involved
        meta_doc_count=D_logical,
        fname="logical_pad.png",
    )

    r = VPMImageReader(path)
    assert r.D == D_phys
    assert r.D_logical == D_logical

    # Export raw u16 so we can do exact equality checks
    rep = r.to_json(include_data=True,
                    data_mode="raw_u16",
                    include_channels="RGB",
                    max_docs=None,
                    downsample=None,
                    pretty=False)

    # Verify doc_indices reflect logical width only
    doc_idx = rep["data"]["doc_indices"]
    assert doc_idx == list(range(D_logical)), "JSON export must trim to logical width"

    # Compare each metric row channel-by-channel to the actual image pixels
    for m, row_entry in enumerate(rep["data"]["rows"]):
        # Raw pixels from the reader, trimmed to logical columns
        raw = r.get_metric_row_raw(m)[:D_logical, :]  # (D_logical, 3) u16

        # R channel
        R_json = row_entry.get("R")
        if R_json is not None:
            assert R_json == raw[:, 0].tolist()

        # G channel
        G_json = row_entry.get("G")
        if G_json is not None:
            assert G_json == raw[:, 1].tolist()

        # B channel
        B_json = row_entry.get("B")
        if B_json is not None:
            assert B_json == raw[:, 2].tolist()

    # For completeness: if we ask normalized, values should equal u16/65535.0
    rep_norm = r.to_json(include_data=True,
                         data_mode="normalized",
                         include_channels="RGB",
                         max_docs=D_logical)
    for m, row_entry in enumerate(rep_norm["data"]["rows"]):
        raw = r.get_metric_row_raw(m)[:D_logical, :].astype(np.float64)
        if "R" in row_entry:
            np.testing.assert_allclose(row_entry["R"], raw[:, 0] / 65535.0, rtol=0, atol=1e-12)
        if "G" in row_entry:
            np.testing.assert_allclose(row_entry["G"], raw[:, 1] / 65535.0, rtol=0, atol=1e-12)
        if "B" in row_entry:
            np.testing.assert_allclose(row_entry["B"], raw[:, 2] / 65535.0, rtol=0, atol=1e-12)
