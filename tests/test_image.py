import numpy as np
from zeromodel.vpm.encoder import VPMEncoder
from zeromodel.utils import to_png_bytes, png_to_gray_array
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_image_order():
    # 1) tiny docs×metrics with a unique max at top-left (doc=0, col=0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 1.0

    # 2) Encode to raw image, then wrap into a real PNG
    img_raw = VPMEncoder(8).encode(M)      # returns array or raw pixel bytes
    png_bytes = to_png_bytes(img_raw)      # <-- important: make it a valid PNG

    # 3) Decode and check orientation
    G = png_to_gray_array(png_bytes)       # (H, W) uint8
    y, x = np.unravel_index(np.argmax(G), G.shape)

    logger.info(f"argmax at: {y}, {x}")  # expect 0, 0 if no vertical flip
    assert (y, x) == (0, 0), f"Expected hottest at top-left, got {(y, x)}"
