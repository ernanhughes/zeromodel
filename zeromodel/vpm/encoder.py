"""VPM (Visual Policy Map) encoding utilities.

This module contains the VPMEncoder class which handles:
- Conversion of normalized, spatially-organized score matrices into RGB image tensors
- Padding of metric channels to 3-channel pixels
- Precision conversion (uint8/uint16/float16/float32/float64)
- Extraction of critical top-left tiles as compact byte payloads

The encoder operates purely on pre-processed numpy arrays and is decoupled from
data sources, normalization pipelines, and storage systems.
"""

import logging
from typing import Optional

import numpy as np

from zeromodel.constants import PRECISION_DTYPE_MAP
from zeromodel.vpm.logic import denormalize_vpm

logger = logging.getLogger(__name__)


class VPMEncoder:
    """
    Stateless encoder for converting decision matrices into visual representations.

    Transforms 2D score matrices (documents × metrics) into 3D image tensors
    (documents × width × 3) where each pixel represents three metrics. This visual
    encoding enables:
    - Efficient storage of decision state
    - Visual interpretation of metric relationships
    - Critical region extraction for fast analysis

    The encoder supports various output precisions for different use cases:
    - uint8/uint16: For visualization and storage efficiency
    - float16/32/64: For precise analytical processing

    Attributes:
        default_output_precision (str): Default precision from config
    """

    def __init__(self, default_output_precision: str = "float32"):
        """Initialize encoder with configuration-based defaults."""
        self.default_output_precision = default_output_precision
        logger.debug(
            "VPMEncoder initialized with default output precision: %s",
            self.default_output_precision,
        )

    def encode(
        self, sorted_matrix: np.ndarray, output_precision: Optional[str] = None
    ) -> np.ndarray:
        """
        Convert a normalized score matrix into a VPM image tensor.

        Process:
        1. Validate input matrix
        2. Determine output precision
        3. Pad metrics dimension to multiple of 3
        4. Reshape to 3D tensor (documents × width × 3)
        5. Convert to target precision and range

        Args:
            sorted_matrix: 2D array of shape (documents, metrics) with values in [0,1]
            output_precision: Target precision (None uses default)

        Returns:
            3D image tensor of shape (documents, ceil(metrics/3), 3)

        Raises:
            ValueError: On invalid input dimensions
        """
        # --- Input Validation ---
        if sorted_matrix is None:
            raise ValueError("Input matrix cannot be None.")
        if sorted_matrix.ndim != 2:
            raise ValueError(f"Matrix must be 2D, got {sorted_matrix.ndim} dimensions.")
        n_docs, n_metrics = sorted_matrix.shape
        if n_docs == 0 or n_metrics == 0:
            raise ValueError("Matrix cannot have zero documents or metrics.")

        # --- Precision Handling ---
        final_precision = output_precision or self.default_output_precision
        if final_precision not in PRECISION_DTYPE_MAP:
            logger.warning(
                "Unsupported precision '%s'. Using default '%s'.",
                final_precision,
                self.default_output_precision,
            )
            final_precision = self.default_output_precision
        target_dtype = PRECISION_DTYPE_MAP[final_precision]

        # --- Matrix Preparation ---
        # Ensure float32 for consistent processing
        matrix = sorted_matrix.astype(np.float32, copy=False)

        # Pad metrics to multiple of 3 (each pixel = 3 metrics)
        padding = (3 - (n_metrics % 3)) % 3  # Calculate padding needed
        if padding:
            matrix = np.pad(
                matrix, ((0, 0), (0, padding)), mode="constant", constant_values=0.0
            )

        # --- Reshaping ---
        width = (n_metrics + padding) // 3
        try:
            # Reshape to 3D tensor: (documents, width, 3)
            img_data = matrix.reshape(n_docs, width, 3)
        except ValueError as e:
            raise ValueError(
                f"Reshape failed: {matrix.shape} → ({n_docs}, {width}, 3)"
            ) from e

        try:
            img = denormalize_vpm(img_data, output_type=target_dtype)
        except ImportError:
            # Fallback conversion
            if target_dtype == np.uint8:
                img = np.clip(img_data * 255.0, 0, 255).astype(target_dtype)
            elif target_dtype == np.uint16:
                img = np.clip(img_data * 65535.0, 0, 65535).astype(target_dtype)
            else:  # Floating point types
                img = np.clip(img_data, 0.0, 1.0).astype(target_dtype)

        logger.debug(
            "Encoded VPM: shape=%s dtype=%s precision=%s",
            img.shape,
            img.dtype,
            final_precision,
        )
        return img

    def get_critical_tile(
        self,
        sorted_matrix: np.ndarray,
        tile_size: int = 3,
        precision: Optional[str] = None,
    ) -> bytes:
        """
            Extract critical top-left tile as compact byte payload.

        The critical tile represents the most important region of the decision
        matrix (highest-ranked documents and metrics).

        Header format (new):
        - Byte 0: width LSB
        - Byte 1: width MSB
        - Byte 2: height LSB
        - Byte 3: height MSB

        Followed by the tile data (flattened array in the selected precision).

            Args:
                sorted_matrix: 2D array of shape (documents, metrics)
                tile_size: Number of documents/pixels to include
                precision: Target data precision

            Returns:
                Byte payload containing header + tile data

            Raises:
                ValueError: On invalid input or tile_size
        """
        # --- Input Validation ---
        if sorted_matrix is None:
            raise ValueError("Input matrix cannot be None.")
        if tile_size <= 0:
            raise ValueError("Tile size must be positive.")
        n_docs, n_metrics = sorted_matrix.shape
        if n_docs == 0 or n_metrics == 0:
            raise ValueError("Matrix cannot have zero documents or metrics.")

        # --- Precision Handling ---
        final_precision = precision or self.default_output_precision
        if final_precision not in PRECISION_DTYPE_MAP:
            logger.warning(
                "Unsupported precision '%s'. Using default '%s'.",
                final_precision,
                self.default_output_precision,
            )
            final_precision = self.default_output_precision
        target_dtype = PRECISION_DTYPE_MAP[final_precision]

        # --- Tile Extraction ---
        # Calculate actual tile dimensions
        actual_h = min(tile_size, n_docs)  # Number of document rows
        tile_metrics_w = min(tile_size * 3, n_metrics)  # Number of metrics
        pixel_w = (tile_metrics_w + 2) // 3  # Resulting pixel width

        # Extract top-left tile
        tile_slice = sorted_matrix[:actual_h, :tile_metrics_w].astype(
            np.float32, copy=False
        )

        # --- Precision Conversion ---
        try:
            # Attempt optimized conversion
            from .logic import denormalize_vpm, normalize_vpm

            tile_norm = normalize_vpm(tile_slice)
            tile_converted = denormalize_vpm(tile_norm, output_type=target_dtype)
        except ImportError:
            # Fallback conversion
            if target_dtype == np.uint8:
                tile_converted = np.clip(tile_slice * 255.0, 0, 255).astype(
                    target_dtype
                )
            elif target_dtype == np.uint16:
                tile_converted = np.clip(tile_slice * 65535.0, 0, 65535).astype(
                    target_dtype
                )
            else:  # Floating point types
                tile_converted = np.clip(tile_slice, 0.0, 1.0).astype(target_dtype)

        # --- Payload Construction ---
        payload = bytearray()
        # New 4-byte header: 16-bit little-endian width and height
        # [0]=width LSB, [1]=width MSB, [2]=height LSB, [3]=height MSB
        payload.append(pixel_w & 0xFF)
        payload.append((pixel_w >> 8) & 0xFF)
        payload.append(actual_h & 0xFF)
        payload.append((actual_h >> 8) & 0xFF)
        # Tile data
        payload.extend(tile_converted.flatten().tobytes())

        logger.debug(
            "Critical tile: size=%d actual=(%d docs, %d px) precision=%s bytes=%d",
            tile_size,
            actual_h,
            pixel_w,
            final_precision,
            len(payload),
        )
        return bytes(payload)

__all__ = ["VPMEncoder"]