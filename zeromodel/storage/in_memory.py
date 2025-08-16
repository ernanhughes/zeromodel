
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from zeromodel.storage.base import StorageBackend


class InMemoryStorage(StorageBackend[np.ndarray]):
    """In-memory storage backend for testing and small datasets."""
    
    def __init__(self):
        self.tiles: Dict[str, np.ndarray] = {}
        self.indices: Dict[int, Dict[str, Any]] = {}
    
    def store_tile(self, level: int, x: int, y: int, data: np.ndarray) -> str:
        tile_id = self.get_tile_id(level, x, y)
        self.tiles[tile_id] = data
        return tile_id
    
    def load_tile(self, tile_id: str) -> Optional[np.ndarray]:
        return self.tiles.get(tile_id)
    
    def query_region(self, level: int, x_start: int, y_start: int, 
                    x_end: int, y_end: int) -> List[Tuple[int, int, np.ndarray]]:
        results = []
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                tile_id = self.get_tile_id(level, x, y)
                tile = self.load_tile(tile_id)
                if tile is not None:
                    results.append((x, y, tile))
        return results
    
    def create_index(self, level: int, index_type: str = "spatial") -> None:
        # In-memory, we don't need explicit indexing
        if level not in self.indices:
            self.indices[level] = {"type": index_type}
