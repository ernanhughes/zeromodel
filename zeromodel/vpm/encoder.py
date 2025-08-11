"""VPM (Visual Policy Map) encoding utilities.

This module contains the VPMEncoder class which is responsible for:
 - Converting a normalized, spatially-organized score matrix into an RGB image tensor
 - Handling padding of metric channels to 3-channel pixels
 - Converting to requested output precision (uint8/uint16/float16/float32/float64)
 - Extracting a critical top-left tile as a compact byte payload

It deliberately knows nothing about DuckDB, feature engineering, or normalization
pipelines; it operates purely on already-prepared numpy arrays.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from zeromodel.config import get_config
from zeromodel.constants import precision_dtype_map

logger = logging.getLogger(__name__)


class VPMEncoder:
    """Stateless encoder for turning normalized matrices into VPM images/tiles."""
    def __init__(self):
        self.default_output_precision = get_config("core").get("default_output_precision", "float32")
        logger.debug("VPMEncoder initialized with default output precision: %s", self.default_output_precision)

    def encode(self, sorted_matrix: np.ndarray, output_precision: Optional[str] = None) -> np.ndarray:
        if sorted_matrix is None:
            raise ValueError("sorted_matrix cannot be None.")
        if sorted_matrix.ndim != 2:
            raise ValueError(f"sorted_matrix must be 2D, got shape {sorted_matrix.shape}.")
        n_docs, n_metrics = sorted_matrix.shape
        if n_docs == 0 or n_metrics == 0:
            raise ValueError("sorted_matrix cannot have zero docs or metrics.")
        final_precision = output_precision or self.default_output_precision
        if final_precision not in precision_dtype_map:
            logger.warning("Unsupported output_precision '%s'. Using default '%s'.", final_precision, self.default_output_precision)
            final_precision = self.default_output_precision
        target_dtype = precision_dtype_map[final_precision]
        matrix = sorted_matrix.astype(np.float32, copy=False)
        width = (n_metrics + 2) // 3
        padding = (3 - (n_metrics % 3)) % 3
        if padding:
            matrix = np.pad(matrix, ((0, 0), (0, padding)), mode='constant', constant_values=0.0)
        try:
            img_data = matrix.reshape(n_docs, width, 3)
        except ValueError as e:
            raise ValueError(f"Cannot reshape data of shape {matrix.shape} to ({n_docs}, {width}, 3).") from e
        try:
            from .logic import denormalize_vpm  # local import to avoid cycle
            img = denormalize_vpm(img_data, output_type=target_dtype)
        except Exception:
            if target_dtype == np.uint8:
                img = np.clip(img_data * 255.0, 0, 255).astype(target_dtype)
            elif target_dtype == np.uint16:
                img = np.clip(img_data * 65535.0, 0, 65535).astype(target_dtype)
            else:
                img = np.clip(img_data, 0.0, 1.0).astype(target_dtype)
        logger.debug("Encoded VPM image: shape=%s dtype=%s (precision=%s)", img.shape, img.dtype, final_precision)
        return img

    def get_critical_tile(self, sorted_matrix: np.ndarray, tile_size: int = 3, precision: Optional[str] = None) -> bytes:
        if sorted_matrix is None:
            raise ValueError("sorted_matrix cannot be None.")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        n_docs, n_metrics = sorted_matrix.shape
        if n_docs == 0 or n_metrics == 0:
            raise ValueError("sorted_matrix cannot have zero docs or metrics.")
        final_precision = precision or self.default_output_precision
        if final_precision not in precision_dtype_map:
            logger.warning("Unsupported tile precision '%s'. Using default '%s'.", final_precision, self.default_output_precision)
            final_precision = self.default_output_precision
        target_dtype = precision_dtype_map[final_precision]
        actual_h = min(tile_size, n_docs)
        tile_metrics_w = min(tile_size * 3, n_metrics)
        pixel_w = (tile_metrics_w + 2) // 3
        tile_slice = sorted_matrix[:actual_h, :tile_metrics_w].astype(np.float32, copy=False)
        try:
            from .logic import denormalize_vpm, normalize_vpm
            tile_norm = normalize_vpm(tile_slice)
            tile_converted = denormalize_vpm(tile_norm, output_type=target_dtype)
        except Exception:
            if target_dtype == np.uint8:
                tile_converted = np.clip(tile_slice * 255.0, 0, 255).astype(target_dtype)
            elif target_dtype == np.uint16:
                tile_converted = np.clip(tile_slice * 65535.0, 0, 65535).astype(target_dtype)
            else:
                tile_converted = np.clip(tile_slice, 0.0, 1.0).astype(target_dtype)
        payload = bytearray()
        payload.append(pixel_w & 0xFF)
        payload.append(actual_h & 0xFF)
        payload.append(0)
        payload.append(0)
        payload.extend(tile_converted.flatten().tobytes())
        logger.debug("Extracted critical tile: tile_size=%d actual=(%d,%d) precision=%s bytes=%d", tile_size, actual_h, pixel_w, final_precision, len(payload))
        return bytes(payload)

__all__ = ["VPMEncoder"]
