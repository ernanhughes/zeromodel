#  zeromodel/provenance/vpf_manager.py
"""
VPFManager — tiny, surgical PNG metadata manager for ZeroModel VPM images.

What it does (no magic, no side effects):
- Reads/writes two iTXt fields in a PNG: "vpf.header" and "vpf.footer".
- Encodes/decodes them as JSON (UTF-8). No custom binary chunks needed.
- Works whether headers/footers are present or not (idempotent helpers).
- Avoids touching pixel data when you only update metadata.

Why iTXt?
- It's part of the PNG spec, UTF-8 friendly, widely ignored by viewers (good),
  and easily accessible via Pillow (PIL).

Public API (all operate on a file path or a PIL Image):
- load_vpf(path) -> (PIL.Image.Image, header: dict, footer: dict)
- save_with_vpf(img_or_array, path, header: dict | None, footer: dict | None)
- read_header(path) / read_footer(path)
- write_header(path, header, inplace=True)
- write_footer(path, footer, inplace=True)
- ensure_header_footer(path, default_header=None, default_footer=None, inplace=True)
- update_header(path, patch: dict, inplace=True)
- update_footer(path, patch: dict, inplace=True)
- has_header(path) / has_footer(path)

All writes use iTXt keys:
    VPF_HEADER_KEY = "vpf.header"
    VPF_FOOTER_KEY = "vpf.footer"

If you ever need compressed text: switch to add_text(..., zip=True) to emit zTXt,
or add a tiny codec (gzip/base64) around the JSON string. Keeping it simple here.

Surgical behavior:
- Reading never mutates.
- Writing rewrites just the PNG container with new text chunks (Pillow path).
- No reliance on your runtime graph; you can unit test it in isolation.

"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, PngImagePlugin

# ---- Constants --------------------------------------------------------------

VPF_HEADER_KEY = "vpf.header"
VPF_FOOTER_KEY = "vpf.footer"

# For consumers that prefer a stable schema, we declare optional shape:
DEFAULT_HEADER_VERSION = "1.0"
DEFAULT_FOOTER_VERSION = "1.0"


# ---- Helpers ----------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_image(img_or_array: Union[Image, "np.ndarray"]) -> Image.Image:
    """Accept a PIL Image or a numpy array and return a PIL Image (no copy if already Image)."""
    if isinstance(img_or_array, Image.Image):
        return img_or_array
    if not isinstance(img_or_array, np.ndarray):
        raise TypeError("img_or_array must be a PIL.Image.Image or a numpy.ndarray")
    mode = "L"
    if img_or_array.ndim == 3:
        if img_or_array.shape[2] == 3:
            mode = "RGB"
        elif img_or_array.shape[2] == 4:
            mode = "RGBA"
    return Image.fromarray(img_or_array, mode=mode)


def _read_itxt(im: Image.Image) -> Dict[str, str]:
    """
    Pillow exposes textual metadata in both im.text (preferred) and im.info.
    We merge them conservatively—im.text wins for duplicate keys.
    """
    text = {}
    # Newer Pillow: .text exists and aggregates iTXt/tEXt/zTXt
    if hasattr(im, "text") and isinstance(im.text, dict):
        text.update(im.text)
    # Fallback: some keys may only appear in .info
    if hasattr(im, "info") and isinstance(im.info, dict):
        for k, v in im.info.items():
            if isinstance(v, str) and k not in text:
                text[k] = v
    return text


def _json_load_or_empty(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        # Corrupt or non-JSON field: return empty to be resilient
        return {}


def _json_dump(d: Optional[Dict[str, Any]]) -> str:
    if not d:
        return "{}"
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def _build_pnginfo(
    header_json: Optional[str], footer_json: Optional[str]
) -> PngImagePlugin.PngInfo:
    """
    Build a PngInfo with our iTXt fields. We use add_itxt to force iTXt (UTF-8).
    """
    meta = PngImagePlugin.PngInfo()
    if header_json is not None:
        meta.add_itxt(VPF_HEADER_KEY, header_json)
    if footer_json is not None:
        meta.add_itxt(VPF_FOOTER_KEY, footer_json)
    return meta


def _rewrite_with_text(
    im: Image.Image,
    out_path: str,
    header: Optional[Dict[str, Any]],
    footer: Optional[Dict[str, Any]],
) -> None:
    """
    Re-save image with (possibly updated) iTXt chunks.
    NOTE: This rewrites the PNG container. Pixel data stays the same source unless PIL re-encodes.
    """
    header_json = _json_dump(header) if header is not None else None
    footer_json = _json_dump(footer) if footer is not None else None
    meta = _build_pnginfo(header_json, footer_json)
    # Preserve mode and transparency where possible
    params = {}
    if "transparency" in im.info:
        params["transparency"] = im.info["transparency"]
    im.save(out_path, format="PNG", pnginfo=meta, **params)


# ---- Data classes (optional schema) -----------------------------------------


@dataclass
class VPFHeader:
    version: str = DEFAULT_HEADER_VERSION
    created_at: str = field(default_factory=_now_iso)
    generator: str = "zeromodel.vpf"
    # user-defined / task-specific (optional)
    task: Optional[str] = None  # e.g., the exact SQL or task string
    order_by: Optional[str] = None  # e.g., "metric1 DESC"
    metric_names: Optional[list[str]] = None
    doc_order: Optional[list[int]] = None  # full 0-based order of docs (top-first)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "generator": self.generator,
            "task": self.task,
            "order_by": self.order_by,
            "metric_names": self.metric_names,
            "doc_order": self.doc_order,
        }


@dataclass
class VPFFooter:
    version: str = DEFAULT_FOOTER_VERSION
    updated_at: str = field(default_factory=_now_iso)
    # navigation/decision result (optional)
    top_docs: Optional[list[int]] = None  # ties allowed: list of doc indices
    relevance_scores: Optional[list[float]] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "top_docs": self.top_docs,
            "relevance_scores": self.relevance_scores,
            "notes": self.notes,
        }


# ---- Public API --------------------------------------------------------------


class VPFManager:
    """
    Minimal, explicit manager for VPF PNG metadata (header/footer).
    All file operations are opt-in. No global state.
    """

    header_key: str = VPF_HEADER_KEY
    footer_key: str = VPF_FOOTER_KEY

    # --- Read ---------------------------------------------------------------

    @staticmethod
    def load_vpf(path: str) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any]]:
        """
        Open a PNG and return (image, header_dict, footer_dict).
        Missing or malformed fields yield {} for that part.
        """
        im = Image.open(path)
        text = _read_itxt(im)
        header = _json_load_or_empty(text.get(VPF_HEADER_KEY))
        footer = _json_load_or_empty(text.get(VPF_FOOTER_KEY))
        return im, header, footer

    @staticmethod
    def read_header(path: str) -> Dict[str, Any]:
        _, header, _ = VPFManager.load_vpf(path)
        return header

    @staticmethod
    def read_footer(path: str) -> Dict[str, Any]:
        _, _, footer = VPFManager.load_vpf(path)
        return footer

    @staticmethod
    def has_header(path: str) -> bool:
        return bool(VPFManager.read_header(path))

    @staticmethod
    def has_footer(path: str) -> bool:
        return bool(VPFManager.read_footer(path))

    # --- Write (file-path oriented) ----------------------------------------

    @staticmethod
    def save_with_vpf(
        img_or_array: Union[Image.Image, "np.ndarray"],
        path: str,
        header: Optional[Dict[str, Any]] = None,
        footer: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a new PNG with provided header/footer dicts.
        If header/footer are None, the corresponding field is omitted.
        """
        im = _to_image(img_or_array)
        _rewrite_with_text(im, path, header, footer)

    @staticmethod
    def write_header(
        path: str,
        header: Dict[str, Any],
        inplace: bool = True,
        out_path: Optional[str] = None,
    ) -> str:
        """
        Write/replace the header in a PNG. Returns the output path.
        """
        im, _, footer = VPFManager.load_vpf(path)
        dst = path if inplace else (out_path or _derive_out_path(path, suffix="_hdr"))
        _rewrite_with_text(im, dst, header, footer if footer else None)
        return dst

    @staticmethod
    def write_footer(
        path: str,
        footer: Dict[str, Any],
        inplace: bool = True,
        out_path: Optional[str] = None,
    ) -> str:
        """
        Write/replace the footer in a PNG. Returns the output path.
        """
        im, header, _ = VPFManager.load_vpf(path)
        dst = path if inplace else (out_path or _derive_out_path(path, suffix="_ftr"))
        _rewrite_with_text(im, dst, header if header else None, footer)
        return dst

    @staticmethod
    def ensure_header_footer(
        path: str,
        default_header: Optional[Dict[str, Any]] = None,
        default_footer: Optional[Dict[str, Any]] = None,
        inplace: bool = True,
        out_path: Optional[str] = None,
    ) -> str:
        """
        Ensure both header and footer exist. If missing, fill with defaults.
        If present, leave untouched. Returns the output path.
        """
        im, header, footer = VPFManager.load_vpf(path)
        new_header = header if header else (default_header or VPFHeader().to_dict())
        new_footer = footer if footer else (default_footer or VPFFooter().to_dict())
        # If nothing to change, optionally return path early
        if header and footer and inplace:
            return path
        dst = path if inplace else (out_path or _derive_out_path(path, suffix="_vpf"))
        _rewrite_with_text(im, dst, new_header, new_footer)
        return dst

    @staticmethod
    def update_header(
        path: str,
        patch: Dict[str, Any],
        inplace: bool = True,
        out_path: Optional[str] = None,
    ) -> str:
        """
        Shallow-merge a patch into the existing header dict.
        Missing header becomes the patch itself.
        """
        im, header, footer = VPFManager.load_vpf(path)
        header = {**header, **patch} if header else dict(patch)
        dst = (
            path if inplace else (out_path or _derive_out_path(path, suffix="_hdr_upd"))
        )
        _rewrite_with_text(im, dst, header, footer if footer else None)
        return dst

    @staticmethod
    def update_footer(
        path: str,
        patch: Dict[str, Any],
        inplace: bool = True,
        out_path: Optional[str] = None,
    ) -> str:
        """
        Shallow-merge a patch into the existing footer dict.
        Missing footer becomes the patch itself.
        """
        im, header, footer = VPFManager.load_vpf(path)
        footer = {**footer, **patch} if footer else dict(patch)
        dst = (
            path if inplace else (out_path or _derive_out_path(path, suffix="_ftr_upd"))
        )
        _rewrite_with_text(im, dst, header if header else None, footer)
        return dst

    # --- Write (in-memory oriented) ----------------------------------------

    @staticmethod
    def to_bytes_with_vpf(
        img_or_array: Union[Image.Image, "np.ndarray"],
        header: Optional[Dict[str, Any]] = None,
        footer: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """Return PNG bytes containing the given header/footer."""
        im = _to_image(img_or_array)
        header_json = _json_dump(header) if header is not None else None
        footer_json = _json_dump(footer) if footer is not None else None
        meta = _build_pnginfo(header_json, footer_json)
        buf = io.BytesIO()
        im.save(buf, format="PNG", pnginfo=meta)
        return buf.getvalue()

    @staticmethod
    def from_bytes(
        png_bytes: bytes,
    ) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any]]:
        """Open PNG bytes and return (image, header, footer)."""
        im = Image.open(io.BytesIO(png_bytes))
        text = _read_itxt(im)
        header = _json_load_or_empty(text.get(VPF_HEADER_KEY))
        footer = _json_load_or_empty(text.get(VPF_FOOTER_KEY))
        return im, header, footer


# ---- internal ---------------------------------------------------------------

def _derive_out_path(path: str, *, suffix: str) -> str:
    if "." in path:
        base, ext = path.rsplit(".", 1)
        return f"{base}{suffix}.{ext}"
    return f"{path}{suffix}.png"

