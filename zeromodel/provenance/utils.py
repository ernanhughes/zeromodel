"""
Utility functions for provenance module.

"It's just a PNG with a tiny header. Survives image pipelines, is easy to cache and diff."
"""

from typing import Dict, Any
import zlib

def compress_vpf_data(data: Dict[str, Any]) -> bytes:
    """Compress VPF data for embedding."""
    json_str = str(data)
    return zlib.compress(json_str.encode('utf-8'))

def decompress_vpf_data(compressed: bytes) -> Dict[str, Any]:
    """Decompress VPF data from embedding."""
    try:
        json_str = zlib.decompress(compressed).decode('utf-8')
        return eval(json_str)  # In practice, use json.loads
    except:
        return {}