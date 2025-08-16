import json
import logging
import math
import struct
import zlib
from typing import (Any, Callable, Dict, Generic, List, Optional, Tuple,
                    TypeVar, Union)

# Create a logger for this module
logger = logging.getLogger(__name__)

T = TypeVar('T')

class StorageBackend(Generic[T]):
    """Abstract interface for storage backends handling world-scale data."""
    
    def store_tile(self, level: int, x: int, y: int, data: T) -> str:
        """Store a tile and return its unique identifier."""
        raise NotImplementedError
    
    def load_tile(self, tile_id: str) -> Optional[T]:
        """Load a tile by its identifier, or return None if not found."""
        raise NotImplementedError
    
    def query_region(self, level: int, x_start: int, y_start: int, 
                    x_end: int, y_end: int) -> List[Tuple[int, int, T]]:
        """Query tiles in a specific rectangular region of a level."""
        raise NotImplementedError
    
    def create_index(self, level: int, index_type: str = "spatial") -> None:
        """Create index for efficient navigation at this level."""
        pass
    
    def get_tile_id(self, level: int, x: int, y: int) -> str:
        """Generate a consistent tile ID for the given coordinates."""
        return f"L{level}_X{x}_Y{y}"
