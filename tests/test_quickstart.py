# quickstart.py — minimal “install, run, see it” demo (instrumented)
import os
import time
import math
import logging
import numpy as np
from PIL import Image
import imageio.v2 as imageio
from zeromodel import ZeroModel, get_critical_tile
from zeromodel.metadata import MetadataView

# -------- logging setup --------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
)
logger = logging.getLogger("quickstart")

class Timer:
    def __init__(self, name):
        self.name = name
        self.t0 = None
        self.elapsed = 0.0
    def __enter__(self):
        self.t0 = time.perf_counter()
        logger.debug(f"[TIMER] {self.name} started")
        return self
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.t0
        logger.info(f"[TIMER] {self.name}: {self.elapsed:.3f}s")

# -------- utility: humanize bytes --------
def human_bytes(n):
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"

# --- 1) Generate 1,000 pronounceable, SQL-safe metric names ---
def gen_metric_names(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    consonants = list("bcdfghjklmnpqrstvwxz")
    vowels = list("aeiou")
    def word():
        syls = rng.integers(2, 5)
        parts = []
        for _ in range(syls):
            c1, v = rng.choice(consonants), rng.choice(vowels)
            if rng.random() < 0.5:
                parts.append(c1 + v)
            else:
                c2 = rng.choice(consonants)
                parts.append(c1 + v + c2)
        return "".join(parts)[:12]
    names, seen = [], set()
    while len(names) < n:
        w = f"{word()}_{word()}" if rng.random() < 0.3 else word()
        key = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in w.lower())
        if key[0].isdigit():
            key = "m_" + key
        if key not in seen:
            seen.add(key)
            names.append(key)
    return names

def test_quickstart():
    profile = {}

    with Timer("generate_metric_names"):
        METRICS = gen_metric_names(1000)
        logger.debug(f"Generated {len(METRICS)} metrics; sample: {METRICS[:5]}")

    PRIMARY_METRIC = METRICS[0]
    SECONDARY_METRIC = METRICS[1]

    num_docs = int(os.getenv("NUM_DOCS", "2048"))
    rng = np.random.default_rng(0)

    with Timer("build_score_matrix"):
        scores = np.zeros((num_docs, len(METRICS)), dtype=np.float32)
        scores[:, 0] = rng.random(num_docs)              # uncertainty
        scores[:, 1] = rng.normal(0.5, 0.15, num_docs)   # size
        scores[:, 2] = rng.random(num_docs) ** 2         # quality (skewed)
        scores[:, 3] = rng.random(num_docs)              # novelty
        scores[:, 4] = 1.0 - scores[:, 0]                # coherence ~ inverse of uncertainty
        est_bytes = scores.nbytes
        logger.info(f"score_matrix shape={scores.shape} dtype={scores.dtype} "
                    f"~{human_bytes(est_bytes)}")

    # Optional: add structure to make ORDER BY visually obvious at scale
    with Timer("inject_structure_for_order_by"):
        row_idx = np.arange(num_docs)
        band_center = num_docs // 3
        band_width  = max(1, num_docs // 10)
        scores[:, 0] = np.clip(
            0.2 + np.exp(-((row_idx - band_center) ** 2) / (2 * (band_width ** 2))).astype(np.float32),
            0, 1
        )
        scores[:, 1] = np.clip(np.linspace(0, 1, num_docs) + rng.normal(0, 0.05, num_docs), 0, 1)
        logger.debug("Injected Gaussian band into PRIMARY metric and gradient+noise into SECONDARY.")

    with Timer("initialize_ZeroModel"):
        zm = ZeroModel(METRICS)

    out_png = os.getenv(os.getcwd(), "images/vpm_demo.png")
    sql = f"SELECT * FROM virtual_index ORDER BY {PRIMARY_METRIC} DESC, {SECONDARY_METRIC} ASC"
    logger.info(f"ORDER BY: {PRIMARY_METRIC} DESC, {SECONDARY_METRIC} ASC")

    with Timer("ZeroModel.prepare(encode->PNG)"):
        zm.prepare(
            score_matrix=scores,
            sql_query=sql,
            nonlinearity_hint=None,
            vpm_output_path=out_png,
        )
    if os.path.exists(out_png):
        size_bytes = os.path.getsize(out_png)
        logger.info(f"VPM written → {os.path.abspath(out_png)} ({human_bytes(size_bytes)})")
        profile["png_size"] = size_bytes
    else:
        logger.error("Expected output PNG not found!")
        return

    with Timer("PIL.verify_png"):
        Image.open(out_png).verify()
        logger.info("PNG verified")

    with Timer("read_metadata"):
        mv = MetadataView.from_png(out_png)
        pretty = mv.pretty()
        # Keep logs compact at INFO; dump full JSON only at DEBUG
        logger.info("Metadata parsed "
                    f"(provenance={'present' if mv.provenance else 'none'}, "
                    f"vpm={'present' if mv.vpm else 'none'})")
        logger.debug("Metadata pretty JSON:\n" + pretty)

    with Timer("read_vpm_pixels_for_critical_tile"):
        vpm_rgb = imageio.imread(out_png)  # H x W x 3
        logger.debug(f"VPM pixel array shape={vpm_rgb.shape} dtype={vpm_rgb.dtype}")
        tile_bytes = get_critical_tile(vpm_rgb, tile_size=3)
        logger.info(f"Critical tile bytes: {len(tile_bytes)}")

    # -------- profiling summary --------
    logger.info("==== Profiling Summary ====")
    for k, v in profile.items():
        logger.info(f"{k}: {human_bytes(v)}")
    logger.info("Done.")

def test_quickstart_compare():
    profile = {}
    os.makedirs("images", exist_ok=True)

    with Timer("generate_metric_names"):
        METRICS = gen_metric_names(2048)

    PRIMARY_METRIC = METRICS[0]
    SECONDARY_METRIC = METRICS[1]

    num_docs = 2048
    rng = np.random.default_rng(0)

    with Timer("build_score_matrix"):
        scores = np.zeros((num_docs, len(METRICS)), dtype=np.float32)
        scores[:, 0] = rng.random(num_docs)              # uncertainty
        scores[:, 1] = rng.normal(0.5, 0.15, num_docs)   # size
        scores[:, 2] = rng.random(num_docs) ** 2         # quality (skewed)
        scores[:, 3] = rng.random(num_docs)              # novelty
        scores[:, 4] = 1.0 - scores[:, 0]                # coherence
        logger.info("score_matrix shape=%s dtype=%s ~%s",
                    scores.shape, scores.dtype, human_bytes(scores.nbytes))

    # make ORDER BY signal obvious
    with Timer("inject_structure_for_order_by"):
        row_idx = np.arange(num_docs)
        band_center = num_docs // 3
        band_width  = max(1, num_docs // 10)
        scores[:, 0] = np.clip(
            0.2 + np.exp(-((row_idx - band_center) ** 2) / (2 * (band_width ** 2))).astype(np.float32),
            0, 1
        )
        scores[:, 1] = np.clip(np.linspace(0, 1, num_docs) + rng.normal(0, 0.05, num_docs), 0, 1)

    # ---------- Case A: DuckDB SQL ----------
    sql_query = f"SELECT * FROM virtual_index ORDER BY {PRIMARY_METRIC} DESC, {SECONDARY_METRIC} ASC"
    os.environ["ZM_USE_DUCKDB"] = "1"
    with Timer("initialize_ZeroModel[sql]"):
        zm_sql = ZeroModel(METRICS)

    out_sql = "images/vpm_sql.png"
    logger.info("ORDER BY (DuckDB): %s DESC, %s ASC", PRIMARY_METRIC, SECONDARY_METRIC)
    with Timer("ZeroModel.prepare[sql]") as t:
        zm_sql.prepare(score_matrix=scores, sql_query=sql_query, vpm_output_path=out_sql)
    profile["prepare_sql_s"] = t.elapsed

    # ---------- Case B: Memory ORDER BY (no DuckDB) ----------
    mem_spec = f"{PRIMARY_METRIC} DESC, {SECONDARY_METRIC} ASC"  # same semantics, no SELECT
    os.environ["ZM_USE_DUCKDB"] = "0"
    with Timer("initialize_ZeroModel[mem]"):
        zm_mem = ZeroModel(METRICS)

    out_mem = "images/vpm_mem.png"
    logger.info("ORDER BY (memory): %s", mem_spec)
    with Timer("ZeroModel.prepare[mem]") as t:
        zm_mem.prepare(score_matrix=scores, sql_query=mem_spec, vpm_output_path=out_mem)
    profile["prepare_mem_s"] = t.elapsed

    # ---------- Case C: No organization (identity) ----------
    os.environ["ZM_USE_DUCKDB"] = "0"
    with Timer("initialize_ZeroModel[none]"):
        zm_none = ZeroModel(METRICS)

    out_none = "images/vpm_none.png"
    logger.info("ORDER BY: none (identity)")
    with Timer("ZeroModel.prepare[none]") as t:
        zm_none.prepare(score_matrix=scores, sql_query=None, vpm_output_path=out_none)
    profile["prepare_none_s"] = t.elapsed

    # ---------- Basic file checks ----------
    for path in (out_sql, out_mem, out_none):
        assert os.path.exists(path), f"Expected PNG not found: {path}"

    # ---------- Equivalence check: SQL vs Memory ordering ----------
    # With continuous floats the chance of exact ties is tiny; they should match.
    assert np.array_equal(zm_sql.doc_order, zm_mem.doc_order), "SQL and memory ORDER BY produced different doc orders"
    assert np.array_equal(zm_sql.metric_order, zm_mem.metric_order), "Metric order should be identical in both modes"

    # ---------- Verify one PNG + read metadata + critical tile ----------
    with Timer("PIL.verify_png"):
        Image.open(out_sql).verify()
        logger.info("PNG verified (sql)")

    with Timer("read_metadata"):
        mv = MetadataView.from_png(out_sql)
        logger.info("Metadata parsed (provenance=%s, vpm=%s)",
                    "present" if mv.provenance else "none",
                    "present" if mv.vpm else "none")

    with Timer("read_vpm_pixels_for_critical_tile"):
        vpm_rgb = imageio.imread(out_sql)
        tile_bytes = get_critical_tile(vpm_rgb, tile_size=3)
        logger.info("Critical tile bytes: %d", len(tile_bytes))

    # -------- profiling summary --------
    logger.info("==== Profiling Summary ====")
    for k, v in profile.items():
        logger.info("%s: %.3fs", k, v)
    logger.info("Done.")
