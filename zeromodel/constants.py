#  zeromodel/constants.py
from __future__ import annotations
import numpy as np

PRECISION_DTYPE_MAP = {
    # Numeric precision values (user-friendly)

    4: np.uint8,
    8: np.uint8,
    16: np.float16,
    32: np.float32,
    64: np.float64,
    # String aliases (for API flexibility)
    "4": np.uint8,
    "8": np.uint8,
    "16": np.float16,
    "32": np.float32,
    "64": np.float64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}

_PNG_SIG = b"\x89PNG\r\n\x1a\n"

DATA_NOT_PROCESSED_ERR = "Data not processed yet. Call process() or prepare() first."
VPM_IMAGE_NOT_READY_ERR = "VPM image not ready. Call prepare() first."
