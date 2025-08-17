"""Utilities for reading and writing PNG text chunks.

This module provides minimal helpers to insert, remove and query text chunks
within PNG images without relying on external libraries.
"""

import struct
import zlib
from typing import Iterator, Optional, Tuple

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _crc32(chunk_type: bytes, data: bytes) -> int:
    return zlib.crc32(chunk_type + data) & 0xffffffff


def _build_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", _crc32(chunk_type, data))
    )


def _iter_chunks(png: bytes) -> Iterator[Tuple[bytes, int, int, bytes]]:
    """Iterate over PNG chunks.

    Yields tuples of (type, start_offset, end_offset, data). ``start_offset``
    points to the 4-byte length field of the chunk and ``end_offset`` points
    right after the CRC (i.e. the start of the next chunk).
    """
    if not png.startswith(_PNG_SIG):
        raise ValueError("Not a PNG: bad signature")
    i = len(_PNG_SIG)
    n = len(png)
    while i + 8 <= n:
        length = struct.unpack(">I", png[i : i + 4])[0]
        ctype = png[i + 4 : i + 8]
        data_start = i + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > n:
            break  # Truncated/corrupt; stop parsing gracefully
        data = png[data_start:data_end]
        yield ctype, i, crc_end, data
        i = crc_end
        if ctype == b"IEND":
            break


def _find_iend_offset(png: bytes) -> int:
    # Return byte offset where IEND chunk starts; insert before this
    for ctype, start, _end, _data in _iter_chunks(png):
        if ctype == b"IEND":
            return start
    raise ValueError("PNG missing IEND chunk")


def _remove_text_chunks_with_key(png: bytes, key: str) -> bytes:
    """Remove existing iTXt/tEXt/zTXt chunks that match ``key``."""
    key_bytes = key.encode("latin-1", "ignore")
    out = bytearray()
    out.extend(png[: len(_PNG_SIG)])
    pos = len(_PNG_SIG)
    for ctype, start, end, data in _iter_chunks(png):
        out.extend(png[pos:start])  # normally empty
        keep = True
        if ctype in (b"tEXt", b"iTXt", b"zTXt"):
            try:
                nul = data.find(b"\x00")
                k = data[:nul] if nul != -1 else b""
            except Exception:
                k = b""
            if k == key_bytes:
                keep = False
        if keep:
            out.extend(png[start:end])
        pos = end
    out.extend(png[pos:])
    return bytes(out)


def _encode_text_chunk(key: str, text: str, use_itxt: bool = True, *, compress: bool = False) -> bytes:
    """Build a tEXt or iTXt chunk."""
    if use_itxt:
        # iTXt layout:
        # keyword\0 compression_flag(1) compression_method(1)
        # language_tag\0 translated_keyword\0 text(UTF-8)
        keyword = key.encode("latin-1", "ignore")[:79]
        comp_flag = b"\x01" if compress else b"\x00"
        comp_method = b"\x00"  # zlib
        language_tag = b""  # empty
        translated_keyword = b""  # empty
        text_bytes = text.encode("utf-8", "strict")
        if compress:
            text_bytes = zlib.compress(text_bytes)
        data = (
            keyword
            + b"\x00"
            + comp_flag
            + comp_method
            + language_tag
            + b"\x00"
            + translated_keyword
            + b"\x00"
            + text_bytes
        )
        return _build_chunk(b"iTXt", data)
    else:
        # tEXt: keyword\0 text (both Latin-1)
        keyword = key.encode("latin-1", "ignore")[:79]
        text_bytes = text.encode("latin-1", "replace")
        data = keyword + b"\x00" + text_bytes
        return _build_chunk(b"tEXt", data)


def _decode_text_chunk(ctype: bytes, data: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Returns (keyword, text) from a tEXt/iTXt/zTXt chunk."""
    try:
        if ctype == b"tEXt":
            nul = data.find(b"\x00")
            if nul == -1:
                return (None, None)
            key = data[:nul].decode("latin-1", "ignore")
            txt = data[nul + 1 :].decode("latin-1", "ignore")
            return key, txt
        if ctype == b"iTXt":
            p = 0
            nul = data.find(b"\x00", p)
            if nul == -1:
                return (None, None)
            key = data[p:nul]
            p = nul + 1
            comp_flag = data[p]
            p += 1
            comp_method = data[p]
            p += 1
            nul = data.find(b"\x00", p)
            if nul == -1:
                return (None, None)
            lang = data[p:nul]
            p = nul + 1
            nul = data.find(b"\x00", p)
            if nul == -1:
                return (None, None)
            trkw = data[p:nul]
            p = nul + 1
            txt_bytes = data[p:]
            if comp_flag == 1:
                txt_bytes = zlib.decompress(txt_bytes)
            key = key.decode("latin-1", "ignore")
            txt = txt_bytes.decode("utf-8", "ignore")
            return key, txt
        if ctype == b"zTXt":
            nul = data.find(b"\x00")
            if nul == -1 or len(data) < nul + 2:
                return (None, None)
            key = data[:nul].decode("latin-1", "ignore")
            comp_method = data[nul + 1]
            comp = data[nul + 2 :]
            if comp_method != 0:
                return key, None
            txt = zlib.decompress(comp).decode("latin-1", "ignore")
            return key, txt
    except Exception:
        pass
    return (None, None)


def png_read_text_chunk(png_bytes: bytes, key: str) -> Optional[str]:
    """Read the text value for ``key`` from iTXt/tEXt/zTXt chunks."""
    if not png_bytes.startswith(_PNG_SIG):
        raise ValueError("Not a PNG")
    want_key = key
    found_text = None
    itxt_text = None
    for ctype, _s, _e, data in _iter_chunks(png_bytes):
        if ctype in (b"tEXt", b"iTXt", b"zTXt"):
            k, v = _decode_text_chunk(ctype, data)
            if k == want_key and v is not None:
                if ctype == b"iTXt":
                    itxt_text = v  # prefer iTXt
                elif found_text is None:
                    found_text = v
    return itxt_text if itxt_text is not None else found_text


def png_write_text_chunk(
    png_bytes: bytes,
    key: str,
    text: str,
    *,
    use_itxt: bool = True,
    compress: bool = False,
    replace_existing: bool = True,
) -> bytes:
    """Insert (or replace) a text chunk with ``(key, text)``.

    ``use_itxt`` selects the iTXt chunk (UTF-8 capable). ``compress`` applies
    zlib compression to the iTXt payload. ``replace_existing`` removes any
    prior chunks with the same key.
    """
    if not png_bytes.startswith(_PNG_SIG):
        raise ValueError("Not a PNG")
    png2 = _remove_text_chunks_with_key(png_bytes, key) if replace_existing else png_bytes
    new_chunk = _encode_text_chunk(key, text, use_itxt=use_itxt, compress=compress)
    iend_off = _find_iend_offset(png2)
    out = png2[:iend_off] + new_chunk + png2[iend_off:]
    return out


__all__ = ["png_read_text_chunk", "png_write_text_chunk"]

