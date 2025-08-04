"""
Hierarchical Edge Device Protocol

This module provides the communication protocol for edge devices
to interact with hierarchical visual policy maps.
"""

import struct
from typing import Any, Dict, Tuple

from .edge import EdgeProtocol


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
    MSG_TYPE_ZOOM = 0x04
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
            level: Hierarchical level to start with
        
        Returns:
            Binary request message
        """
        # Format: [version][type][level][task_length][task_bytes]
        task_bytes = task_description.encode('utf-8')
        if len(task_bytes) > 253:
            task_bytes = task_bytes[:253]  # Truncate if too long
        
        return struct.pack(
            f"BBBB{len(task_bytes)}s",
            HierarchicalEdgeProtocol.PROTOCOL_VERSION,
            HierarchicalEdgeProtocol.MSG_TYPE_REQUEST,
            level,
            len(task_bytes),
            task_bytes
        )
    
    @staticmethod
    def parse_tile(tile_data: bytes) -> Tuple[int, int, int, int, int, bytes]:
        """
        Parse a tile message from the proxy.
        
        Args:
            tile_ Binary tile data
        
        Returns:
            (level, width, height, x_offset, y_offset, pixels)
        """
        if len(tile_data) < 5:
            raise ValueError("Invalid tile format: too short")
        
        level = tile_data[0]
        width = tile_data[1]
        height = tile_data[2]
        x_offset = tile_data[3]
        y_offset = tile_data[4]
        pixels = tile_data[5:]
        
        # Validate dimensions
        if width > HierarchicalEdgeProtocol.MAX_TILE_WIDTH:
            width = HierarchicalEdgeProtocol.MAX_TILE_WIDTH
        if height > HierarchicalEdgeProtocol.MAX_TILE_HEIGHT:
            height = HierarchicalEdgeProtocol.MAX_TILE_HEIGHT
        
        return level, width, height, x_offset, y_offset, pixels
    
    @staticmethod
    def make_decision(tile_data: bytes) -> bytes:
        """
        Process a tile and make a decision.
        
        Args:
            tile_ Binary tile data from parse_tile()
        
        Returns:
            Binary decision message
        """
        # Parse the tile
        level, width, height, x, y, pixels = HierarchicalEdgeProtocol.parse_tile(tile_data)
        
        # Simple decision logic: check top-left pixel value
        top_left_value = pixels[0] if len(pixels) > 0 else 128
        
        # Decision: is this "dark enough" to be relevant?
        is_relevant = 1 if top_left_value < 128 else 0
        
        # Create decision message
        # Format: [version][type][level][decision][reserved]
        return struct.pack("BBBBB", 
                          HierarchicalEdgeProtocol.PROTOCOL_VERSION,
                          HierarchicalEdgeProtocol.MSG_TYPE_DECISION,
                          level,
                          is_relevant,
                          0)  # Reserved byte
    
    @staticmethod
    def request_zoom(tile_data: bytes, direction: str = "in") -> bytes:
        """
        Request to zoom in or out from current position.
        
        Args:
            tile_ Binary tile data
            direction: "in" or "out"
        
        Returns:
            Binary zoom request message
        """
        # Parse the tile to get current level
        level, _, _, _, _, _ = HierarchicalEdgeProtocol.parse_tile(tile_data)
        
        # Determine new level
        new_level = level
        if direction == "in":
            new_level = max(0, level - 1)  # Level 0 is most abstract
        elif direction == "out":
            new_level = min(2, level + 1)  # Assuming max 3 levels
        
        # Create zoom message
        # Format: [version][type][current_level][new_level]
        return struct.pack("BBBB", 
                          HierarchicalEdgeProtocol.PROTOCOL_VERSION,
                          HierarchicalEdgeProtocol.MSG_TYPE_ZOOM,
                          level,
                          new_level)
    
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
            HierarchicalEdgeProtocol.PROTOCOL_VERSION,
            HierarchicalEdgeProtocol.MSG_TYPE_ERROR,
            code,
            msg_bytes
        )