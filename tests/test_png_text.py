import importlib.util
import io
from pathlib import Path

from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "png_text", Path(__file__).resolve().parents[1] / "zeromodel" / "png_text.py"
)
png_text = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(png_text)

png_read_text_chunk = png_text.png_read_text_chunk
png_write_text_chunk = png_text.png_write_text_chunk
_iter_chunks = png_text._iter_chunks


def _blank_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=0).save(buf, format="PNG")
    return buf.getvalue()


def test_itxt_round_trip_and_replacement():
    png = _blank_png()
    png2 = png_write_text_chunk(png, "desc", "hello")
    assert png_read_text_chunk(png2, "desc") == "hello"
    # ensure only one text chunk exists
    assert (
        sum(1 for ctype, *_ in _iter_chunks(png2) if ctype in (b"tEXt", b"iTXt", b"zTXt"))
        == 1
    )
    # replace with compressed version
    png3 = png_write_text_chunk(png2, "desc", "bye", compress=True)
    assert png_read_text_chunk(png3, "desc") == "bye"
    assert (
        sum(1 for ctype, *_ in _iter_chunks(png3) if ctype in (b"tEXt", b"iTXt", b"zTXt"))
        == 1
    )


def test_text_chunk():
    png = _blank_png()
    png2 = png_write_text_chunk(png, "k", "latin", use_itxt=False)
    assert png_read_text_chunk(png2, "k") == "latin"
