# zeromodel/hierarchical_edge.py
"""
Hierarchical Edge Device Protocol

This module provides the communication protocol for edge devices
to interact with hierarchical visual policy maps.
"""

import logging
import struct
from typing import Tuple

logger = logging.getLogger(__name__)


class HierarchicalEdgeProtocol:
    """
    Protocol for edge devices to interact with hierarchical VPMs.

    This implements a minimal protocol that:
    - Works with tiny memory constraints (<25KB)
    - Handles hierarchical navigation
    - Enables zero-model intelligence at the edge
    """

    # Protocol version (1 byte)
    PROTOCOL_VERSION = 1

    # Message types (1 byte each)
    MSG_TYPE_REQUEST = 0x01
    MSG_TYPE_TILE = 0x02
    MSG_TYPE_DECISION = 0x03
    MSG_TYPE_ZOOM = 0x04  # Request or indication to change hierarchical level
    MSG_TYPE_ERROR = 0x05

    # Maximum tile size (for memory constraints)
    MAX_TILE_WIDTH = 3
    MAX_TILE_HEIGHT = 3

    @staticmethod
    def create_request(task_description: str, level: int = 0) -> bytes:
        """
        Create a request message for the edge proxy.

        Args:
            task_description: Natural language task description
            level: Starting hierarchical level (0 = most abstract)

        Returns:
            Binary request message

        Raises:
            ValueError: If task_description is None or level is invalid.
        """
        logger.debug(
            f"Creating request for task: '{task_description}', starting level: {level}"
        )
        if task_description is None:
            error_msg = "Task description cannot be None"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not (0 <= level <= 255):  # Assuming 1-byte level
            error_msg = f"Level must be between 0 and 255, got {level}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        task_bytes = task_description.encode("utf-8")
        original_len = len(task_bytes)
        max_task_len = 253  # 256 - 3 bytes for version, type, level
        if original_len > max_task_len:
            logger.warning(
                f"Task description ({original_len} bytes) exceeds {max_task_len} bytes, truncating."
            )
            task_bytes = task_bytes[:max_task_len]  # Truncate if too long
        elif original_len == 0:
            logger.info("Creating request with an empty task description.")

        message = struct.pack(
            f"BBBB{len(task_bytes)}s",
            HierarchicalEdgeProtocol.PROTOCOL_VERSION,
            HierarchicalEdgeProtocol.MSG_TYPE_REQUEST,
            level & 0xFF,  # Ensure level fits in 1 byte
            len(task_bytes),
            task_bytes,
        )
        logger.debug(f"Request message created, size: {len(message)} bytes")
        return message

    @staticmethod
    def parse_tile(tile_data: bytes) -> Tuple[int, int, int, int, int, bytes]:
        """
        Parse a tile message from the proxy.

        Args:
            tile_data: Binary tile data
                          Format: [level][width][height][x_offset][y_offset][...pixels...]

        Returns:
            Tuple containing (level, width, height, x_offset, y_offset, pixels_data)

        Raises:
            ValueError: If tile_data is invalid (e.g., too short).
        """
        header_size = 5  # level + width + height + x_offset + y_offset
        logger.debug(f"Parsing tile data, received size: {len(tile_data)} bytes")
        if not tile_data or len(tile_data) < header_size:
            error_msg = f"Invalid tile format: data too short (expected >= {header_size} bytes, got {len(tile_data)})."
            logger.error(error_msg)
            raise ValueError(error_msg)

        level = tile_data[0]
        width = tile_data[1]
        height = tile_data[2]
        x_offset = tile_data[3]
        y_offset = tile_data[4]
        pixels_data = tile_data[header_size:]  # Remaining bytes are pixel data

        logger.debug(
            f"Parsed tile header: level={level}, width={width}, height={height}, x_offset={x_offset}, y_offset={y_offset}"
        )

        # Validate dimensions strictly based on protocol limits
        if width > HierarchicalEdgeProtocol.MAX_TILE_WIDTH:
            logger.warning(
                f"Tile width ({width}) exceeds MAX_TILE_WIDTH ({HierarchicalEdgeProtocol.MAX_TILE_WIDTH}). Processing with max width."
            )
            width = (
                HierarchicalEdgeProtocol.MAX_TILE_WIDTH
            )  # Original logic commented for clarity
        if height > HierarchicalEdgeProtocol.MAX_TILE_HEIGHT:
            logger.warning(
                f"Tile height ({height}) exceeds MAX_TILE_HEIGHT ({HierarchicalEdgeProtocol.MAX_TILE_HEIGHT}). Processing with max height."
            )
            height = (
                HierarchicalEdgeProtocol.MAX_TILE_HEIGHT
            )  # Original logic commented for clarity

        result = (level, width, height, x_offset, y_offset, pixels_data)
        logger.debug(
            f"Tile parsed successfully: {result[:5]} + {len(pixels_data)} pixel bytes"
        )
        return result

    @staticmethod
    def make_decision(tile_message_data: bytes) -> bytes:
        """
        Process a tile message and make a decision.
        Assumes the tile_message_data includes the full tile message structure
        starting with level, width, height, x_offset, y_offset.

        Args:
            tile_message_data: Binary tile message data (payload).

        Returns:
            Binary decision message (MSG_TYPE_DECISION).
        """
        logger.debug(
            f"Making decision based on tile data ({len(tile_message_data)} bytes)"
        )
        try:
            level, width, height, x_offset, y_offset, pixels_data = (
                HierarchicalEdgeProtocol.parse_tile(tile_message_data)
            )
        except ValueError as e:
            logger.error(f"Failed to parse tile for decision making: {e}")
            # Return an error message instead of raising
            return HierarchicalEdgeProtocol.create_error(
                11, f"Tile Parse Error: {str(e)}"
            )

        # Simple decision logic: check top-left pixel value (R channel)
        # Check if we have at least one pixel's R channel data
        # Pixel data is [R0, G0, B0, R1, G1, B1, ...]
        # Top-left pixel R is at index 0
        is_relevant = 0  # Default to not relevant
        if len(pixels_data) > 0:
            top_left_r_value = pixels_data[0]
            # Decision: is this "dark enough" to be relevant?
            is_relevant = 1 if top_left_r_value < 128 else 0
            logger.debug(
                f"Top-left pixel R value: {top_left_r_value}, Decision (is_relevant): {is_relevant}"
            )
        else:
            logger.warning(
                "Tile pixel data is empty. Defaulting decision to not relevant."
            )
            # is_relevant remains 0

        # Create decision message
        # Format: [version][type][level][decision][reserved]
        decision_message = struct.pack(
            "BBBBB",
            HierarchicalEdgeProtocol.PROTOCOL_VERSION,
            HierarchicalEdgeProtocol.MSG_TYPE_DECISION,
            level & 0xFF,  # Ensure level fits in 1 byte
            is_relevant,
            0,
        )  # Reserved byte
        logger.info(
            f"Decision made at level {level}: {'RELEVANT' if is_relevant else 'NOT RELEVANT'}"
        )
        logger.debug(f"Decision message created, size: {len(decision_message)} bytes")
        return decision_message

    @staticmethod
    def request_zoom(tile_message_data: bytes, direction: str = "in") -> bytes:
        """
        Request to zoom in or out from the level indicated in the current tile.

        Args:
            tile_message_data: Binary tile message data (payload) to get the current level.
            direction: "in" (towards detail) or "out" (towards abstraction).

        Returns:
            Binary zoom request message (MSG_TYPE_ZOOM).

        Raises:
            ValueError: If direction is invalid or tile data cannot be parsed.
        """
        logger.debug(f"Creating zoom request: direction='{direction}'")
        if direction not in ["in", "out"]:
            error_msg = f"Invalid zoom direction '{direction}'. Must be 'in' or 'out'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            current_level, _, _, _, _, _ = HierarchicalEdgeProtocol.parse_tile(
                tile_message_data
            )
        except ValueError as e:
            logger.error(
                f"Failed to parse tile to determine current level for zoom: {e}"
            )
            # Return an error message instead of raising
            return HierarchicalEdgeProtocol.create_error(
                12, f"Zoom Parse Error: {str(e)}"
            )

        # Determine new level
        # Assuming Level 0 = Most Abstract, Level 2 = Most Detailed (common convention)
        new_level = current_level
        if direction == "in":
            # Zooming in means going to a more detailed level (higher number)
            new_level = min(2, current_level + 1)  # Assuming max 3 levels (0, 1, 2)
            logger.debug(f"Zoom IN requested: Level {current_level} -> {new_level}")
        elif direction == "out":
            # Zooming out means going to a more abstract level (lower number)
            new_level = max(0, current_level - 1)
            logger.debug(f"Zoom OUT requested: Level {current_level} -> {new_level}")

        # Create zoom message
        # Format: [version][type][current_level][new_level]
        zoom_message = struct.pack(
            "BBBB",
            HierarchicalEdgeProtocol.PROTOCOL_VERSION,
            HierarchicalEdgeProtocol.MSG_TYPE_ZOOM,
            current_level & 0xFF,  # Ensure level fits in 1 byte
            new_level & 0xFF,  # Ensure level fits in 1 byte
        )
        logger.debug(f"Zoom request message created, size: {len(zoom_message)} bytes")
        return zoom_message

    @staticmethod
    def create_error(code: int, message: str = "") -> bytes:
        """
        Create an error message.

        Args:
            code: Error code (1 byte recommended)
            message: Optional error message (max ~252 chars)

        Returns:
            Binary error message (MSG_TYPE_ERROR)
        """
        logger.debug(f"Creating error message: code={code}, message='{message}'")
        if code < 0 or code > 255:  # Assuming 1-byte code
            logger.warning(
                f"Error code {code} is outside typical 0-255 range for 1-byte field."
            )
        # Max message length considering 3 header bytes (Version, Type, Code)
        max_msg_len = 253
        msg_bytes = message.encode("utf-8")[:max_msg_len]

        error_message = struct.pack(
            f"BBB{len(msg_bytes)}s",
            HierarchicalEdgeProtocol.PROTOCOL_VERSION,
            HierarchicalEdgeProtocol.MSG_TYPE_ERROR,
            code & 0xFF,  # Ensure code fits in 1 byte
            msg_bytes,
        )
        logger.debug(f"Error message created, size: {len(error_message)} bytes")
        return error_message


# --- Configure logging for this module ---
# This should ideally be done once in your main application.
# Placing it here ensures logs appear if this script is run directly.
if __name__ == "__main__":
    # Example configuration - adjust as needed for your application
    logging.basicConfig(
        level=logging.DEBUG,  # Adjust level (DEBUG, INFO, WARNING, ERROR)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()  # Output to console
            # logging.FileHandler('hierarchical_edge_protocol.log') # Optional: Output to a file
        ],
    )
    logger.info("Logging configured for HierarchicalEdgeProtocol module.")
