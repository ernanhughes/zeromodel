"""
Edge Device Protocol

This module provides the EdgeProtocol class which implements a minimal
protocol for edge devices with <25KB memory. It handles:
- Receiving policy map tiles from a proxy
- Making decisions based on the tile
- Sending back results with minimal overhead
"""

import struct
from typing import Tuple, Dict, Any

class EdgeProtocol:
    """
    Communication protocol for ZeroMI edge devices with <25KB memory.
    
    This implements a minimal protocol that:
    - Works with tiny memory constraints
    - Requires minimal processing
    - Survives network transmission
    - Enables zero-model decision making
    
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
        """
        # Format: [version][type][task_length][task_bytes]
        task_bytes = task_description.encode('utf-8')
        if len(task_bytes) > 255:
            task_bytes = task_bytes[:255]  # Truncate if too long
        
        return struct.pack(
            f"BBB{len(task_bytes)}s",
            EdgeProtocol.PROTOCOL_VERSION,
            EdgeProtocol.MSG_TYPE_REQUEST,
            len(task_bytes),
            task_bytes
        )
    
    @staticmethod
    def parse_tile(tile_data: bytes) -> Tuple[int, int, int, int, bytes]:
        """
        Parse a tile message from the proxy.
        
        Args:
            tile_data: Binary tile data
        
        Returns:
            (width, height, x_offset, y_offset, pixels)
        """
        if len(tile_data) < 4:
            raise ValueError("Invalid tile format: too short")
        
        width = tile_data[0]
        height = tile_data[1]
        x_offset = tile_data[2]
        y_offset = tile_data[3]
        pixels = tile_data[4:]
        
        # Validate dimensions
        if width > EdgeProtocol.MAX_TILE_WIDTH:
            width = EdgeProtocol.MAX_TILE_WIDTH
        if height > EdgeProtocol.MAX_TILE_HEIGHT:
            height = EdgeProtocol.MAX_TILE_HEIGHT
        
        return width, height, x_offset, y_offset, pixels
    
    @staticmethod
    def make_decision(tile_data: bytes) -> bytes:
        """
        Process a tile and make a decision.
        
        Args:
            tile_data: Binary tile data from parse_tile()
        
        Returns:
            Binary decision message
        """
        # Parse the tile
        width, height, x, y, pixels = EdgeProtocol.parse_tile(tile_data)
        
        # Simple decision logic: check top-left pixel value
        # For critical channel (first 3 metrics), we look at R channel of top-left
        top_left_value = pixels[0] if len(pixels) > 0 else 128
        
        # Decision: is this "dark enough" to be relevant?
        is_relevant = 1 if top_left_value < 128 else 0
        
        # Create decision message
        # Format: [version][type][decision][reserved]
        return struct.pack("BBBB", 
                          EdgeProtocol.PROTOCOL_VERSION,
                          EdgeProtocol.MSG_TYPE_DECISION,
                          is_relevant,
                          0)  # Reserved byte
    
    @staticmethod
    def create_error(code: int, message: str = "") -> bytes:
        """
        Create an error message.
        
        Args:
            code: Error code
            message: Optional error message
        
        Returns:
            Binary error message
        """
        msg_bytes = message.encode('utf-8')[:252]  # Leave room for headers
        
        return struct.pack(
            f"BBB{len(msg_bytes)}s",
            EdgeProtocol.PROTOCOL_VERSION,
            EdgeProtocol.MSG_TYPE_ERROR,
            code,
            msg_bytes
        )