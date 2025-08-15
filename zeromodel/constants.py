import numpy as np

precision_dtype_map = {
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
    "float64": np.float64
}