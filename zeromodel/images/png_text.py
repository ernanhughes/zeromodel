# zeromodel/png_text.py
import struct
import zlib
from typing import List, Optional, Tuple

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _crc32(chunk_type: bytes, data: bytes) -> int:
    return zlib.crc32(chunk_type + data) & 0xFFFFFFFF


def _build_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", _crc32(chunk_type, data))
    )


def _iter_chunks(png: bytes) -> List[Tuple[bytes, int, int, bytes]]:
    """
    Yields (type, start_offset, end_offset, data).
    start_offset points to the 4-byte length field of the chunk.
    end_offset points right AFTER the CRC (i.e., start of the next chunk).
    """
    if not png.startswith(_PNG_SIG):
        raise ValueError("Not a PNG: bad signature")
    out = []
    i = len(_PNG_SIG)
    n = len(png)
    while i + 8 <= n:
        if i + 8 > n:
            break
        length = struct.unpack(">I", png[i : i + 4])[0]
        ctype = png[i + 4 : i + 8]
        data_start = i + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > n:
            # Truncated/corrupt; stop parsing gracefully
            break
        data = png[data_start:data_end]
        out.append((ctype, i, crc_end, data))
        i = crc_end
        if ctype == b"IEND":
            break
    return out


def _find_iend_offset(png: bytes) -> int:
    # Return byte offset where IEND chunk starts; insert before this
    for ctype, start, end, _ in _iter_chunks(png):
        if ctype == b"IEND":
            return start
    raise ValueError("PNG missing IEND chunk")


def _remove_text_chunks_with_key(png: bytes, key: str) -> bytes:
    """Remove existing iTXt/tEXt/zTXt chunks that match `key`."""
    key_bytes = key.encode("latin-1", "ignore")
    chunks = _iter_chunks(png)
    pieces = [png[: len(_PNG_SIG)]]
    for ctype, start, end, data in chunks:
        if ctype in (b"tEXt", b"iTXt", b"zTXt"):
            # Parse enough to get the keyword for filtering
            try:
                if ctype == b"tEXt":
                    # keyword\0text (both Latin-1)
                    nul = data.find(b"\x00")
                    k = data[:nul] if nul != -1 else b""
                elif ctype == b"iTXt":
                    # keyword\0compflag\0compmeth\0lang\0trkw\0text
                    # We only need 'keyword'
                    nul = data.find(b"\x00")
                    k = data[:nul] if nul != -1 else b""
                else:  # zTXt (compressed Latin-1 text)
                    nul = data.find(b"\x00")
                    k = data[:nul] if nul != -1 else b""
            except Exception:
                k = b""
            if k == key_bytes:
                # skip (remove)
                continue
        # keep the chunk bytes verbatim
        pieces.append(png[start:end])
    return b"".join(pieces)


def _encode_text_chunk(
    key: str, text: str, use_itxt: bool = True, compress: bool = False
) -> bytes:
    """
    Build a tEXt or iTXt chunk bytes.
    - iTXt supports full UTF-8; we default to iTXt (uncompressed).
    - tEXt requires Latin-1. We'll encode lossy if needed.
    """
    if use_itxt:
        # iTXt layout:
        # keyword\0 compression_flag(1)\0 compression_method(1)\0 language_tag\0 translated_keyword\0 text(UTF-8)
        keyword = key.encode("latin-1", "ignore")[:79]  # spec: 1-79 bytes
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
            + b"\x00"
            + comp_method
            + b"\x00"
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


def _decode_text_chunk(
    ctype: bytes, data: bytes
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (keyword, text) from a tEXt/iTXt/zTXt chunk; unknown/invalid -> (None, None).
    """
    try:
        if ctype == b"tEXt":
            nul = data.find(b"\x00")
            if nul == -1:
                return (None, None)
            key = data[:nul].decode("latin-1", "ignore")
            txt = data[nul + 1 :].decode("latin-1", "ignore")
            return (key, txt)
        elif ctype == b"iTXt":
            # Parse fields up to the text payload
            # keyword\0 comp_flag(1)\0 comp_method(1)\0 language\0 translated\0 text
            p = 0
            nul = data.find(b"\x00", p)
            key = data[p:nul]
            p = nul + 1
            comp_flag = data[p]
            p += 2  # skip comp_flag and the \0
            comp_method = data[p]
            p += 2
            nul = data.find(b"\x00", p)
            lang = data[p:nul]
            p = nul + 1
            nul = data.find(b"\x00", p)
            trkw = data[p:nul]
            p = nul + 1
            txt_bytes = data[p:]
            if comp_flag == 1:  # compressed
                txt_bytes = zlib.decompress(txt_bytes)
            key = key.decode("latin-1", "ignore")
            txt = txt_bytes.decode("utf-8", "ignore")
            return key, txt
        elif ctype == b"zTXt":
            nul = data.find(b"\x00")
            if nul == -1 or len(data) < nul + 2:
                return (None, None)
            key = data[:nul].decode("latin-1", "ignore")
            # data[nul+1] is compression method; payload starts at nul+2
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
    """
    Read the text value for a given key from iTXt/tEXt/zTXt.
    Prefer iTXt if both exist. Returns None if not found.
    """
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
    """
    Insert (or replace) a text chunk with (key, text).
    - use_itxt=True => UTF-8 capable iTXt (recommended)
    - compress=True => compress iTXt payload with zlib
    - replace_existing=True => remove any prior chunks for `key` (tEXt/iTXt/zTXt)
    """
    if not png_bytes.startswith(_PNG_SIG):
        raise ValueError("Not a PNG")
    # Remove existing entries for this key (both tEXt/iTXt/zTXt)
    png2 = (
        _remove_text_chunks_with_key(png_bytes, key) if replace_existing else png_bytes
    )
    # Build new chunk
    new_chunk = _encode_text_chunk(key, text, use_itxt=use_itxt, compress=compress)
    # Insert before IEND
    iend_off = _find_iend_offset(png2)
    out = png2[:iend_off] + new_chunk + png2[iend_off:]
    return out
