# zeromodel/images/core.py
from __future__ import annotations

import pickle
import struct
from zeromodel.provenance.schema import VPF_VERSION, VPF, VPFAuth, VPFDeterminism, VPFInputs, VPFLineage, VPFMetrics, VPFModel, VPFParams, VPFPipeline
from dataclasses import asdict
from typing import Any, Optional, Tuple, Dict, Union

import numpy as np
from PIL import Image

import base64
import hashlib
import json
import zlib
from io import BytesIO
from typing import List


from zeromodel.provenance.metadata import VPF_FOOTER_MAGIC, VPF_MAGIC_HEADER
from zeromodel.provenance.png_text import png_read_text_chunk, png_write_text_chunk


# ========= VPM header (expected by tests) ====================================
_ZMPK_MAGIC = b"ZMPK"  # 4 bytes
# Layout written into the RGB raster (row-major, R then G then B):
#   [ Z M P K ] [ uint32 payload_len ] [ payload bytes ... ]
# payload = pickle.dumps(obj) for arbitrary state

# =============================
# Public API
# =============================

def _vpf_to_dict(obj: Union["VPF", Dict[str, Any]]) -> Dict[str, Any]:
    return asdict(obj) if not isinstance(obj, dict) else obj


def _coerce_vpf(obj: Union[VPF, Dict[str, Any]]) -> VPF:
    """
    Accept either a VPF dataclass or a legacy dict (like the tests use); return a VPF dataclass.
    Maps a few legacy keys (e.g., determinism.seed -> seed_global, inputs.prompt_sha3 -> prompt_hash).
    """
    if isinstance(obj, VPF):
        return obj

    d = dict(obj or {})

    # Pipeline
    p = dict(d.get("pipeline") or {})
    pipeline = VPFPipeline(
        graph_hash=str(p.get("graph_hash", "")),
        step=str(p.get("step", "")),
        step_schema_hash=str(p.get("step_schema_hash", "")),
    )

    # Model
    m = dict(d.get("model") or {})
    model = VPFModel(
        id=str(m.get("id", "")),
        assets=dict(m.get("assets") or {}),
    )

    # Determinism (support legacy "seed")
    det = dict(d.get("determinism") or {})
    seed = det.get("seed", det.get("seed_global", 0))
    determinism = VPFDeterminism(
        seed_global=int(seed or 0),
        seed_sampler=int(det.get("seed_sampler", seed or 0)),
        rng_backends=list(det.get("rng_backends") or []),
    )

    # Params (allow width/height or size)
    par = dict(d.get("params") or {})
    size = par.get("size") or [par.get("width", 0), par.get("height", 0)]
    if not (isinstance(size, (list, tuple)) and len(size) >= 2):
        size = [0, 0]
    params = VPFParams(
        sampler=str(par.get("sampler", "")),
        steps=int(par.get("steps", 0) or 0),
        cfg_scale=float(par.get("cfg_scale", 0.0) or 0.0),
        size=[int(size[0] or 0), int(size[1] or 0)],
        preproc=list(par.get("preproc") or []),
        postproc=list(par.get("postproc") or []),
        stripe=par.get("stripe"),  # tolerate/forward optional metadata
    )

    # Inputs (map prompt_sha3 -> prompt_hash)
    inp = dict(d.get("inputs") or {})
    prompt_hash = inp.get("prompt_hash") or inp.get("prompt_sha3") or ""
    inputs = VPFInputs(
        prompt=str(inp.get("prompt", "")),
        negative_prompt=inp.get("negative_prompt"),
        prompt_hash=str(prompt_hash),
        image_refs=list(inp.get("image_refs") or []),
        retrieved_docs_hash=inp.get("retrieved_docs_hash"),
        task=str(inp.get("task", "")),
    )

    # Metrics
    met = dict(d.get("metrics") or {})
    metrics = VPFMetrics(
        aesthetic=float(met.get("aesthetic", 0.0) or 0.0),
        coherence=float(met.get("coherence", 0.0) or 0.0),
        safety_flag=float(met.get("safety_flag", 0.0) or 0.0),
    )

    # Lineage
    lin = dict(d.get("lineage") or {})
    lineage = VPFLineage(
        parents=list(lin.get("parents") or []),
        content_hash=str(lin.get("content_hash", "")),
        vpf_hash=str(lin.get("vpf_hash", "")),
    )

    # Signature
    sig = d.get("signature")
    signature = None
    if isinstance(sig, dict) and sig:
        signature = VPFAuth(
            algo=str(sig.get("algo", "")),
            pubkey=str(sig.get("pubkey", "")),
            sig=str(sig.get("sig", "")),
        )

    return VPF(
        vpf_version=str(d.get("vpf_version", VPF_VERSION)),
        pipeline=pipeline,
        model=model,
        determinism=determinism,
        params=params,
        inputs=inputs,
        metrics=metrics,
        lineage=lineage,
        signature=signature,
    )


def _vpf_from(obj: Union["VPF", Dict[str, Any]]) -> "VPF":
    return _coerce_vpf(obj)


def create_vpf(
    pipeline: Dict[str, Any],
    model: Dict[str, Any],
    determinism: Dict[str, Any],
    params: Dict[str, Any],
    inputs: Dict[str, Any],
    metrics: Dict[str, Any],
    lineage: Dict[str, Any],
    signature: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Flexible builder: accepts partial dicts (like tests do) and returns a plain dict.
    """
    raw = {
        "vpf_version": VPF_VERSION,
        "pipeline": pipeline or {},
        "model": model or {},
        "determinism": determinism or {},
        "params": params or {},
        "inputs": inputs or {},
        "metrics": metrics or {},
        "lineage": lineage or {},
        "signature": signature or None,
    }
    return _vpf_to_dict(_coerce_vpf(raw))


# --- Hashing / (de)serialization ---------------------------------------------


def _compute_content_hash(data: bytes) -> str:
    return f"sha3:{hashlib.sha3_256(data).hexdigest()}"


def _compute_vpf_hash(vpf_like: Union[VPF, Dict[str, Any]]) -> str:
    d = (
        _vpf_to_dict(_vpf_from(vpf_like))
        if not isinstance(vpf_like, dict)
        else vpf_like
    )
    clean = json.loads(json.dumps(d, sort_keys=True))
    if "lineage" in clean and "vpf_hash" in clean["lineage"]:
        del clean["lineage"]["vpf_hash"]
    payload = json.dumps(clean, sort_keys=True).encode("utf-8")
    return "sha3:" + hashlib.sha3_256(payload).hexdigest()


def _serialize_vpf(vpf: Union[VPF, Dict[str, Any]]) -> bytes:
    dc = _vpf_from(vpf)
    d = asdict(dc)
    d["lineage"]["vpf_hash"] = _compute_vpf_hash(d.copy())
    json_data = json.dumps(d, sort_keys=True).encode("utf-8")
    comp = zlib.compress(json_data)
    return VPF_MAGIC_HEADER + struct.pack(">I", len(comp)) + comp


def _deserialize_vpf(data: bytes) -> VPF:
    if data[:4] != VPF_MAGIC_HEADER:
        raise ValueError("Invalid VPF magic")
    L = struct.unpack(">I", data[4:8])[0]
    comp = data[8 : 8 + L]
    j = json.loads(zlib.decompress(comp))

    # integrity
    expected = _compute_vpf_hash(j)
    if j.get("lineage", {}).get("vpf_hash") != expected:
        raise ValueError("VPF hash mismatch")

    # Be robust against unknown / missing keys in nested dicts
    def _pick(d: Optional[dict], allowed: set) -> dict:
        d = d or {}
        return {k: d[k] for k in d.keys() & allowed}

    _params_allowed = {
        "sampler",
        "steps",
        "cfg_scale",
        "size",
        "preproc",
        "postproc",
        "stripe",
    }
    _inputs_allowed = {
        "prompt",
        "negative_prompt",
        "prompt_hash",
        "image_refs",
        "retrieved_docs_hash",
        "task",
    }
    # Allow common analytics keys in metrics (keeps VPFMetrics small but tolerant)
    _metrics_allowed = {
        "aesthetic",
        "coherence",
        "safety_flag",
        # zeromodel extras we’ve seen:
        "documents",
        "metrics",
        "top_doc_global",
        "relevance",
    }
    _lineage_allowed = {"parents", "content_hash", "vpf_hash", "timestamp"}

    return VPF(
        vpf_version=j.get("vpf_version", VPF_VERSION),
        pipeline=VPFPipeline(**(j.get("pipeline") or {})),
        model=VPFModel(**(j.get("model") or {})),
        determinism=VPFDeterminism(**(j.get("determinism") or {})),
        params=VPFParams(**_pick(j.get("params"), _params_allowed)),
        inputs=VPFInputs(**_pick(j.get("inputs"), _inputs_allowed)),
        metrics=VPFMetrics(**_pick(j.get("metrics"), _metrics_allowed)),
        lineage=VPFLineage(**_pick(j.get("lineage"), _lineage_allowed)),
        signature=VPFAuth(**j["signature"]) if j.get("signature") else None,
    )


# --- Extraction / Embedding ---------------------------------------------------


def extract_vpf_from_png_bytes(png_bytes: bytes) -> tuple[Dict[str, Any], dict]:
    """
    Returns (vpf_dict, meta). Prefers iTXt 'vpf' chunk; falls back to ZMVF footer.
    Always returns a plain dict (not a dataclass) for test compatibility.
    """
    raw = png_read_text_chunk(png_bytes, key="vpf")
    if raw:
        vpf_bytes = base64.b64decode(raw)
        vpf_obj = _deserialize_vpf(vpf_bytes)  # dataclass
        vpf_dict = _vpf_to_dict(vpf_obj)
        return vpf_dict, {"embedding_mode": "itxt", "confidence": 1.0}

    # Fallback: legacy footer (ZMVF + length + zlib(JSON))
    try:
        vpf_dict = read_json_footer(png_bytes)  # already a dict
        return vpf_dict, {"embedding_mode": "footer", "confidence": 0.6}
    except Exception:
        raise ValueError("No embedded VPF found (neither iTXt 'vpf' nor ZMVF footer)")


def extract_vpf(
    obj: Union[bytes, bytearray, memoryview, Image.Image],
) -> tuple[Dict[str, Any], dict]:
    """
    Unified extractor:
      - If `obj` is PNG bytes → parse iTXt 'vpf' (preferred) and return (vpf_dict, metadata)
      - If `obj` is a PIL.Image → serialize to PNG bytes then parse as above
    """
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return extract_vpf_from_png_bytes(bytes(obj))
    if isinstance(obj, Image.Image):
        buf = BytesIO()
        obj.save(buf, format="PNG")
        return extract_vpf_from_png_bytes(buf.getvalue())
    raise TypeError("extract_vpf expects PNG bytes or a PIL.Image")


def _sha3_tagged(data: bytes) -> str:
    return "sha3:" + hashlib.sha3_256(data).hexdigest()


def embed_vpf(
    image: Image.Image,
    vpf: Union[VPF, Dict[str, Any]],
    *,
    add_stripe: Optional[bool] = None,
    compress: bool = False,
    mode: Optional[str] = None,
    stripe_metrics_matrix: Optional[np.ndarray] = None,
    stripe_metric_names: Optional[List[str]] = None,
    stripe_channels: Tuple[str, ...] = ("R",),
) -> bytes:
    """
    Write VPF into PNG:
      1) Optionally paint a tiny header stripe (4 rows) with a magic tag + quickscan metric means.
      2) Serialize the (possibly painted) image to PNG.
      3) Compute content_hash over *core PNG* (no iTXt 'vpf', no ZMVF footer).
      4) Compute vpf_hash over canonical JSON (excluding lineage.vpf_hash).
      5) Store VPF in a 'vpf' iTXt chunk.
      6) Optionally append a legacy ZMVF footer with zlib(JSON) for compatibility.
    Returns PNG bytes.
    """
    vpf_dc = _vpf_from(vpf)  # dataclass
    vpf_dict = _vpf_to_dict(vpf_dc)  # plain dict for JSON

    # Determine if stripe was requested
    stripe_requested = (
        (mode == "stripe") or bool(add_stripe) or (stripe_metrics_matrix is not None)
    )

    img = image.copy()
    if stripe_requested and img.height >= _HEADER_ROWS:
        # Paint the stripe in-place (quickscan means only; robust & cheap)
        try:
            img = _encode_header_stripe(
                img,
                metric_names=stripe_metric_names,
                metrics_matrix=stripe_metrics_matrix,
                channels=stripe_channels,
            )
            # Store a tiny hint into VPF params so tooling knows a stripe exists
            vpf_dict.setdefault("params", {})
            vpf_dict["params"]["stripe"] = {
                "header_rows": _HEADER_ROWS,
                "channels": list(stripe_channels),
                "metric_names": list(stripe_metric_names or []),
                "encoding": "means:v1",
            }
        except Exception:
            # fail open: keep going without stripe
            pass

    # 1) Serialize *image only* to PNG bytes (no VPF yet)
    buf = BytesIO()
    img.save(buf, format="PNG")
    png0 = buf.getvalue()

    # 2) Compute *core* hash (no footer, no iTXt 'vpf')
    core = png_core_bytes(png0)
    vpf_dict.setdefault("lineage", {})
    vpf_dict["lineage"]["content_hash"] = _sha3_tagged(core)

    # 3) Compute/refresh vpf_hash (ignore any existing lineage.vpf_hash)
    vpf_dict_copy = json.loads(json.dumps(vpf_dict, sort_keys=True))
    if "lineage" in vpf_dict_copy:
        vpf_dict_copy["lineage"].pop("vpf_hash", None)
    payload_for_hash = json.dumps(vpf_dict_copy, sort_keys=True).encode("utf-8")
    vpf_dict["lineage"]["vpf_hash"] = (
        "sha3:" + hashlib.sha3_256(payload_for_hash).hexdigest()
    )

    # 4) Write iTXt 'vpf' with canonical container
    vpf_json_sorted = json.dumps(vpf_dict, sort_keys=True).encode("utf-8")
    vpf_comp = zlib.compress(vpf_json_sorted)
    vpf_container = VPF_MAGIC_HEADER + struct.pack(">I", len(vpf_comp)) + vpf_comp
    payload_b64 = base64.b64encode(vpf_container).decode("ascii")
    png_with_itxt = png_write_text_chunk(
        png0,
        key="vpf",
        text=payload_b64,
        use_itxt=True,
        compress=compress,
        replace_existing=True,
    )

    # 5) Legacy ZMVF footer: pure zlib(JSON) for backwards tools/tests
    # Keep behavior controlled by 'mode' or 'add_stripe' (historical)
    if stripe_requested:
        footer_payload = zlib.compress(vpf_json_sorted)
        footer = (
            VPF_FOOTER_MAGIC + len(footer_payload).to_bytes(4, "big") + footer_payload
        )
        return png_with_itxt + footer

    return png_with_itxt


def verify_vpf(vpf: Union[VPF, Dict[str, Any]], artifact_bytes: bytes) -> bool:
    v = _vpf_to_dict(_vpf_from(vpf))
    # 1) content hash over core PNG
    expected = v.get("lineage", {}).get("content_hash", "")
    ok_content = True
    if expected:
        core = png_core_bytes(artifact_bytes)
        ok_content = _sha3_tagged(core) == expected

    # 2) internal vpf_hash
    d = json.loads(json.dumps(v, sort_keys=True))
    if "lineage" in d:
        d["lineage"].pop("vpf_hash", None)
    recomputed = (
        "sha3:"
        + hashlib.sha3_256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()
    )
    ok_vpf = recomputed == v.get("lineage", {}).get("vpf_hash", "")

    return ok_content and ok_vpf


def validate_vpf(
    vpf: Union[VPF, Dict[str, Any]], artifact_bytes: bytes
) -> Dict[str, Any]:
    """
    Validate VPF integrity and return detailed results.

    Returns:
        Dictionary with validation results for each component
    """
    results = {
        "content_hash": False,
        "vpf_hash": False,
        "signature": False,
        "overall": False,
    }

    # Normalize to dict for uniform access
    v = _vpf_to_dict(_vpf_from(vpf))

    # Content hash validation
    expected_ch = v.get("lineage", {}).get("content_hash")
    if expected_ch:
        computed = _sha3_tagged(artifact_bytes)
        results["content_hash"] = computed == expected_ch

    # VPF hash validation
    recomputed = _compute_vpf_hash(v)
    results["vpf_hash"] = recomputed == v.get("lineage", {}).get("vpf_hash")

    # Signature validation (placeholder)
    if v.get("signature"):
        results["signature"] = True

    results["overall"] = all(
        [
            results["content_hash"],
            results["vpf_hash"],
            results["signature"] if v.get("signature") else True,
        ]
    )

    return results


# --- PNG core helpers ---------------------------------------------------------


def _strip_footer(png: bytes) -> bytes:
    """Remove trailing ZMVF footer (if present)."""
    idx = png.rfind(VPF_FOOTER_MAGIC)
    return png if idx == -1 else png[:idx]


def _strip_vpf_itxt(png: bytes) -> bytes:
    """
    Return PNG bytes with any iTXt/tEXt/zTXt chunk whose key is 'vpf' removed.
    Leaves all other chunks intact.
    """
    sig = b"\x89PNG\r\n\x1a\n"
    if not png.startswith(sig):
        return png
    out = bytearray(sig)
    i = len(sig)
    n = len(png)
    while i + 12 <= n:
        length = struct.unpack(">I", png[i : i + 4])[0]
        ctype = png[i + 4 : i + 8]
        data_start = i + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > n:
            break
        chunk = png[i:crc_end]

        if ctype in (b"iTXt", b"tEXt", b"zTXt"):
            data = png[data_start:data_end]
            key = data.split(b"\x00", 1)[0]  # key up to first NUL
            if key == b"vpf":
                i = crc_end
                if ctype == b"IEND":  # extremely unlikely, but bail safely
                    break
                continue

        out.extend(chunk)
        i = crc_end
        if ctype == b"IEND":
            break
    return bytes(out)


def png_core_bytes(png_with_metadata: bytes) -> bytes:
    """
    Core PNG = PNG without our provenance containers:
      - strip ZMVF footer
      - strip iTXt/tEXt/zTXt chunks whose key is 'vpf'
    """
    no_footer = _strip_footer(png_with_metadata)
    return _strip_vpf_itxt(no_footer)


# --- ZeroModel convenience ----------------------------------------------------


def create_vpf_for_zeromodel(
    task: str,
    doc_order: List[int],
    metric_order: List[int],
    total_documents: int,
    total_metrics: int,
    model_id: str = "zero-1.0",
) -> Dict[str, Any]:
    """
    Create a VPF specifically for ZeroModel use cases (returns dict).
    """
    return create_vpf(
        pipeline={
            "graph_hash": f"sha3:{task}",
            "step": "spatial-organization",
            "step_schema_hash": "sha3:zeromodel-v1",
        },
        model={"id": model_id, "assets": {}},
        determinism={"seed_global": 0, "seed_sampler": 0, "rng_backends": ["numpy"]},
        params={"task": task, "doc_order": doc_order, "metric_order": metric_order},
        inputs={"task": task},
        metrics={
            "documents": total_documents,
            "metrics": total_metrics,
            "top_doc_global": doc_order[0] if doc_order else 0,
        },
        lineage={
            "parents": [],
            "content_hash": "",  # Will be filled later
            "vpf_hash": "",  # Will be filled during serialization
        },
    )


def extract_decision_from_vpf(vpf: VPF) -> Tuple[int, Dict[str, Any]]:
    """
    Extract decision information from VPF.

    Returns:
        (top_document_index, decision_metadata)
    """
    metrics = vpf.metrics
    lineage = vpf.lineage

    # Get top document from metrics
    top_doc = getattr(metrics, "top_doc_global", 0)

    # Extract additional decision metadata
    metadata = {
        "confidence": getattr(metrics, "relevance", 1.0),
        "timestamp": getattr(lineage, "timestamp", None),
        "source": "vpf_embedded",
    }

    return (top_doc, metadata)


def merge_vpfs(parent_vpf: VPF, child_vpf: VPF) -> VPF:
    """
    Merge two VPFs, preserving lineage and creating a new parent-child relationship.
    """
    # Create new VPF with combined lineage
    new_vpf = VPF(
        vpf_version=VPF_VERSION,
        pipeline=child_vpf.pipeline,
        model=child_vpf.model,
        determinism=child_vpf.determinism,
        params=child_vpf.params,
        inputs=child_vpf.inputs,
        metrics=child_vpf.metrics,
        lineage=VPFLineage(
            parents=[parent_vpf.lineage.vpf_hash]
            if parent_vpf.lineage.vpf_hash
            else [],
            content_hash=child_vpf.lineage.content_hash,
            vpf_hash="",  # Will be computed during serialization
        ),
        signature=child_vpf.signature,
    )

    return new_vpf


def _hex_to_rgb(seed_hex: str) -> tuple[int, int, int]:
    """
    Map a hex digest to a stable RGB tuple. Uses the first 6, 6, 6 hex digits
    from the digest (cycled if shorter).
    """
    s = (seed_hex or "").lower()
    if s.startswith("sha3:"):
        s = s[5:]
    if not s:
        s = "0000000000000000000000000000000000000000000000000000000000000000"
    # take 3 slices of 2 hex chars each
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)


def replay_from_vpf(
    vpf: Union[VPF, Dict[str, Any]], output_path: Optional[str] = None
) -> bytes:
    v = _vpf_to_dict(_vpf_from(vpf))
    try:
        w, h = (
            int((v.get("params", {}).get("size") or [512, 512])[0]),
            int((v.get("params", {}).get("size") or [512, 512])[1]),
        )
    except Exception:
        w, h = 512, 512
    seed = (
        v.get("inputs", {}).get("prompt_hash")
        or v.get("lineage", {}).get("vpf_hash", "")
    ) or ""
    color = _hex_to_rgb(seed)
    img = Image.new("RGB", (max(1, w), max(1, h)), color=color)
    b = BytesIO()
    img.save(b, format="PNG")
    data = b.getvalue()
    if output_path:
        with open(output_path, "wb") as f:
            f.write(data)
    return data


def read_json_footer(blob: bytes) -> dict:
    """
    Extract JSON from the ZMVF footer: ZMVF | uint32(len) | zlib(JSON)
    Raises ValueError on format errors.
    """
    if not isinstance(blob, (bytes, bytearray)):
        raise TypeError("read_json_footer expects bytes")

    if len(blob) < len(VPF_FOOTER_MAGIC) + 4:
        raise ValueError("Blob too small for footer")

    # Find footer anywhere in blob (not just at end)
    footer_pos = blob.rfind(VPF_FOOTER_MAGIC)
    if footer_pos == -1:
        raise ValueError("Footer magic not found")

    # Validate length fields
    try:
        payload_len = int.from_bytes(blob[footer_pos + 4 : footer_pos + 8], "big")
    except Exception:
        raise ValueError("Invalid length field")

    payload_start = footer_pos + 8
    if payload_start + payload_len > len(blob):
        raise ValueError("Footer extends beyond blob")

    try:
        comp = bytes(blob[payload_start : payload_start + payload_len])
        return json.loads(zlib.decompress(comp).decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to decompress/parse footer: {e}")


# --- Stripe (header) helpers --------------------------------------------------

_HEADER_ROWS = 4  # reserve top 4 rows


def _encode_header_stripe(
    base: Image.Image,
    *,
    metric_names: Optional[List[str]],
    metrics_matrix: Optional[np.ndarray],
    channels: Tuple[str, ...] = ("R",),
) -> Image.Image:
    """
    Paint a tiny 4-row header at the top of the image with:
      - Row 0: ASCII 'VPF1' tag in RGB (for quick detection)
      - Row 1+: up to 3 rows of quickscan metric means in specified channels.
    We only store coarse stats (means) to keep it simple & robust.

    Args:
        base: PIL image (any mode). We'll write into RGB buffer then convert back.
        metric_names: names (K,) aligned with metrics_matrix columns
        metrics_matrix: (Hvals, K) float32 in [0,1] or any numeric (we clamp)
        channels: subset of ("R","G","B") to use; ("R",) means write into red only.

    Returns:
        New PIL.Image with header rows painted.
    """
    if base.height < _HEADER_ROWS:
        return base  # nothing to do

    img = base.convert("RGB").copy()
    arr = np.array(img, dtype=np.uint8)
    H, W, _ = arr.shape

    # Row 0: magic marker "VPF1" across first 4 pixels as ASCII codes.
    magic = b"VPF1"
    for i, bval in enumerate(magic):
        if i < W:
            arr[0, i, :] = 0
            arr[0, i, 0] = bval  # store ASCII in red channel for simplicity

    # Nothing else to write?
    if metrics_matrix is None or metric_names is None or metrics_matrix.size == 0:
        return Image.fromarray(arr, mode="RGB").convert(base.mode)

    # Normalize & compute means per metric (K,)
    m = np.asarray(metrics_matrix, dtype=np.float32)
    if m.ndim == 1:
        m = m[:, None]
    K = m.shape[1]
    means = np.clip(np.nanmean(m, axis=0), 0.0, 1.0)  # clamp to [0,1]

    # Which RGB channels to use
    ch_index = {"R": 0, "G": 1, "B": 2}
    used = [ch_index[c] for c in channels if c in ch_index]
    if not used:
        used = [0]  # default to red

    # Rows 1..3: we can store up to 3 groups of metric means (by channel)
    # We write first min(K, W) metrics into columns left->right as 8-bit values.
    # If multiple channels requested, we replicate the same means into each channel row.
    n_rows_payload = min(_HEADER_ROWS - 1, len(used))
    ncols = min(K, W)
    payload = (means[:ncols] * 255.0 + 0.5).astype(np.uint8)

    for r in range(n_rows_payload):
        row = 1 + r
        ch = used[r]
        arr[row, :ncols, ch] = payload
        # zero other channels on that row (cosmetic, keeps stripe crisp)
        for ch_other in (0, 1, 2):
            if ch_other != ch:
                arr[row, :ncols, ch_other] = 0

    return Image.fromarray(arr, mode="RGB").convert(base.mode)


def _decode_header_stripe(
    png_bytes: bytes,
) -> Dict[str, Any]:
    """
    Quickscan the first 4 rows of the PNG to pull back coarse metric means
    encoded by _encode_header_stripe(). Safe if no stripe is present.
    Returns a dict like:
      {
        "present": bool,
        "rows": 4 or 0,
        "channels": ["R"],  # best guess
        "metric_means_0": [...],  # values in [0,1] from the first payload row
        "metric_means_1": [...],  # if present (2nd payload row), etc.
      }
    """
    try:
        im = Image.open(BytesIO(png_bytes))
        if im.height < _HEADER_ROWS:
            return {"present": False, "rows": 0}
        arr = np.array(im.convert("RGB"), dtype=np.uint8)
    except Exception:
        return {"present": False, "rows": 0}

    H, W, _ = arr.shape
    # Check magic
    magic_ok = (
        W >= 4
        and arr[0, 0, 0] == ord("V")
        and arr[0, 1, 0] == ord("P")
        and arr[0, 2, 0] == ord("F")
        and arr[0, 3, 0] == ord("1")
    )
    if not magic_ok:
        return {"present": False, "rows": 0}

    out = {"present": True, "rows": _HEADER_ROWS, "channels": []}
    # Extract up to 3 payload rows; detect which channel carries values
    for r in range(1, _HEADER_ROWS):
        row = arr[r, :, :]
        # pick the channel with the largest variance as the data channel
        variances = [row[:, ch].var() for ch in (0, 1, 2)]
        ch = int(np.argmax(variances))
        out["channels"].append(["R", "G", "B"][ch])
        # read non-zero prefix as payload (stop when trailing zeros dominate)
        data = row[:, ch]
        # heuristic: read up to last non-zero, but cap length (W)
        nz = np.nonzero(data)[0]
        if nz.size == 0:
            out[f"metric_means_{r-1}"] = []
            continue
        ncols = int(nz[-1]) + 1
        vals = (data[:ncols].astype(np.float32) / 255.0).tolist()
        out[f"metric_means_{r-1}"] = vals
    return out


def tensor_to_vpm(
    tensor: Any,
    min_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """
    Encode ANY Python/NumPy structure into a VPM image (RGB carrier) using
    the ZMPK format expected by the tests.

    Pixel stream layout:
        ZMPK | uint32(len) | payload

    payload = pickle.dumps(tensor, highest protocol)
    """
    payload = pickle.dumps(tensor, protocol=pickle.HIGHEST_PROTOCOL)
    blob = _ZMPK_MAGIC + struct.pack(">I", len(payload)) + payload
    return _bytes_to_rgb_image(blob, min_size=min_size)


def vpm_to_tensor(img: Image.Image) -> Any:
    """
    Decode a VPM image produced by `tensor_to_vpm` back into the object.
    """
    raw = _image_to_bytes(img)
    if len(raw) < 8:
        raise ValueError("VPM too small to contain header")

    magic = bytes(raw[:4])
    if magic != _ZMPK_MAGIC:
        raise ValueError("Bad VPM magic; not a ZMPK-encoded image")

    n = struct.unpack(">I", bytes(raw[4:8]))[0]
    if n < 0 or 8 + n > len(raw):
        raise ValueError("Corrupt VPM length")

    payload = bytes(raw[8 : 8 + n])
    return pickle.loads(payload)


# =============================
# Internal helpers
# =============================


def _bytes_to_rgb_image(
    blob: bytes, *, min_size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    # Find minimum WxH so that W*H*3 >= len(blob)
    total = len(blob)
    side = int(np.ceil(np.sqrt(total / 3.0)))
    w = h = max(16, side)
    if min_size is not None:
        mw, mh = int(min_size[0]), int(min_size[1])
        w = max(w, mw)
        h = max(h, mh)

    arr = np.zeros((h, w, 3), dtype=np.uint8)
    flat = arr.reshape(-1)

    # Fill flat RGB stream with blob
    flat[: min(total, flat.size)] = np.frombuffer(
        blob, dtype=np.uint8, count=min(total, flat.size)
    )
    return Image.fromarray(arr)  # mode inferred from shape/dtype


def _image_to_bytes(img: Image.Image) -> bytearray:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return bytearray(arr.reshape(-1))
