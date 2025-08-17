# zeromodel/edge.py
"""
Edge Device Protocol

This module provides the EdgeProtocol class which implements a minimal
protocol for edge devices with <25KB memory. It handles:
- Receiving policy map tiles from a proxy
- Making decisions based on the tile
- Sending back results with minimal overhead
"""

import logging
import struct
from typing import Tuple

logger = logging.getLogger(__name__)

class EdgeProtocol:
    """
    Communication protocol for zeromodel edge devices with <25KB memory.
    
    This implements a minimal protocol that:
    - Works with tiny memory constraints
    - Requires minimal processing
    - Survives network transmission
    - Enables zero-model decision-making
    
    Designed to work with as little as 180 bytes of code on the device.
    """
    
    # Protocol version (1 byte)
    PROTOCOL_VERSION = 1
    
    # Message types (1 byte each)
    MSG_TYPE_REQUEST = 0x01
    MSG_TYPE_TILE = 0x02
    MSG_TYPE_DECISION = 0x03
    MSG_TYPE_ERROR = 0x04
    
    # Maximum tile size (for memory constraints)
    MAX_TILE_WIDTH = 3
    MAX_TILE_HEIGHT = 3

    @staticmethod
    def create_request(task_description: str) -> bytes:
        """
        Create a request message for the edge proxy.
        
        Args:
            task_description: Natural language task description
        
        Returns:
            Binary request message
            
        Raises:
            ValueError: If task_description is None.
        """
        logger.debug(f"Creating request for task: '{task_description}'")
        if task_description is None:
             error_msg = "Task description cannot be None"
             logger.error(error_msg)
             raise ValueError(error_msg)

        task_bytes = task_description.encode('utf-8')
        original_len = len(task_bytes)
        if original_len > 255:
            logger.warning(f"Task description ({original_len} bytes) exceeds 255 bytes, truncating.")
            task_bytes = task_bytes[:255] # Truncate if too long
        elif original_len == 0:
             logger.info("Creating request with an empty task description.")

        message = struct.pack(
            f"BBB{len(task_bytes)}s",
            EdgeProtocol.PROTOCOL_VERSION,
            EdgeProtocol.MSG_TYPE_REQUEST,
            len(task_bytes), # Length of the task description
            task_bytes
        )
        logger.debug(f"Request message created, size: {len(message)} bytes")
        return message
    
    @staticmethod
    def parse_tile(tile_data: bytes) -> Tuple[int, int, int, int, bytes]:
        """
        Parse a tile message from the proxy.
        
        Args:
            tile_data: Binary tile data (at least 4 bytes header: 16-bit LE width and height)
        
        Returns:
            Tuple containing (width, height, x_offset, y_offset, pixels_data)
            
        Raises:
            ValueError: If tile_data is invalid (e.g., too short, invalid dimensions).
        """
        header_size = 4
        logger.debug(f"Parsing tile data, received size: {len(tile_data)} bytes")
        if not tile_data or len(tile_data) < header_size:
            error_msg = f"Invalid tile format: data too short (expected >= {header_size} bytes, got {len(tile_data)})."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 16-bit little-endian width and height
        width = tile_data[0] | (tile_data[1] << 8)
        height = tile_data[2] | (tile_data[3] << 8)
        x_offset = 0
        y_offset = 0
        pixels_data = tile_data[header_size:]  # Remaining bytes are pixel data

        logger.debug(f"Parsed tile header: width={width}, height={height}, x_offset={x_offset}, y_offset={y_offset}")

        # Validate dimensions strictly based on protocol limits
        # Note: The original logic modified width/height if they exceeded limits,
        # but this can be confusing. It's often better to reject invalid data.
        # However, to maintain potential compatibility with the original intent,
        # we can log warnings but proceed, assuming the *data* conforms to the limits.
        # A stricter approach would be:
        # if not (0 < width <= EdgeProtocol.MAX_TILE_WIDTH) or not (0 < height <= EdgeProtocol.MAX_TILE_HEIGHT):
        #    error_msg = f"Invalid tile dimensions: width={width} (max {EdgeProtocol.MAX_TILE_WIDTH}), height={height} (max {EdgeProtocol.MAX_TILE_HEIGHT})"
        #    logger.error(error_msg)
        #    raise ValueError(error_msg)

        if width > EdgeProtocol.MAX_TILE_WIDTH:
            logger.warning(f"Tile width ({width}) exceeds MAX_TILE_WIDTH ({EdgeProtocol.MAX_TILE_WIDTH}). Processing with max width.")
        if height > EdgeProtocol.MAX_TILE_HEIGHT:
            logger.warning(f"Tile height ({height}) exceeds MAX_TILE_HEIGHT ({EdgeProtocol.MAX_TILE_HEIGHT}). Processing with max height.")

        # Optional: Check if the actual pixel data length matches the expected size
        # expected_pixel_count = width * height * 3 # Assuming 3 channels per pixel
        # if len(pixels_data) < expected_pixel_count:
        #     logger.warning(f"Tile pixel data ({len(pixels_data)} bytes) is shorter than expected ({expected_pixel_count} bytes).")

        result = (width, height, x_offset, y_offset, pixels_data)
        logger.debug(f"Tile parsed successfully: {result[:4]} + {len(pixels_data)} pixel bytes")
        return result
    
    @staticmethod
    def make_decision(tile_message_data: bytes) -> bytes:
        """
        Process a tile message and make a decision.
        Assumes the tile_message_data includes the full message structure,
        including the header bytes if it came directly from a received message.
        If it's just the payload from a MSG_TYPE_TILE message, it should start
        with width, height, x_offset, y_offset.

        For simplicity, and based on the original `parse_tile` logic,
        we'll assume `tile_message_data` is the payload starting with
        width, height, x_offset, y_offset.

        Args:
            tile_message_data: Binary tile data (payload).
        
        Returns:
            Binary decision message (MSG_TYPE_DECISION).
        """
        logger.debug(f"Making decision based on tile data ({len(tile_message_data)} bytes)")
        try:
            _, _, _, _, pixels_data = EdgeProtocol.parse_tile(tile_message_data)
        except ValueError as e:
            logger.error(f"Failed to parse tile for decision making: {e}")
            # Return an error message instead of raising, if the protocol expects messages back
            return EdgeProtocol.create_error(10, f"Tile Parse Error: {str(e)}")

        # Simple decision logic: check top-left pixel value (R channel)
        # Check if we have at least one pixel's R channel data
        # Pixel data is [R0, G0, B0, R1, G1, B1, ...]
        # Top-left pixel R is at index 0
        is_relevant = 0 # Default to not relevant
        if len(pixels_data) > 0:
            top_left_r_value = pixels_data[0]
            # Decision: is this "dark enough" to be relevant?
            is_relevant = 1 if top_left_r_value < 128 else 0
            logger.debug(f"Top-left pixel R value: {top_left_r_value}, Decision (is_relevant): {is_relevant}")
        else:
             logger.warning("Tile pixel data is empty. Defaulting decision to not relevant.")
             # is_relevant remains 0

        # Create decision message
        # Format: [version][type][decision][reserved]
        decision_message = struct.pack("BBBB", 
                          EdgeProtocol.PROTOCOL_VERSION,
                          EdgeProtocol.MSG_TYPE_DECISION,
                          is_relevant,
                          0)  # Reserved byte
        logger.info(f"Decision made: {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")
        logger.debug(f"Decision message created, size: {len(decision_message)} bytes")
        return decision_message
    
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
        if code < 0 or code > 255: # Assuming 1-byte code
             logger.warning(f"Error code {code} is outside typical 0-255 range for 1-byte field.")
        msg_bytes = message.encode('utf-8')[:252]  # Leave room for headers (1+1+1+252 = 255 max)
        
        error_message = struct.pack(
            f"BBB{len(msg_bytes)}s",
            EdgeProtocol.PROTOCOL_VERSION,
            EdgeProtocol.MSG_TYPE_ERROR,
            code & 0xFF, # Ensure code fits in 1 byte
            msg_bytes
        )
        logger.debug(f"Error message created, size: {len(error_message)} bytes")
        return error_message
