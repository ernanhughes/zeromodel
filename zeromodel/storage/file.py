# zeromodel/storage/file.py
import os
from .base import StorageBackend

class FileStorage(StorageBackend):
    """File system storage backend for local deployments."""
    
    def __init__(self, base_dir: str = "vpm_tiles"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def store_tile(self, level: int, x: int, y: int, data: bytes) -> str:
        tile_id = self.get_tile_id(level, x, y)
        path = os.path.join(self.base_dir, f"{tile_id}.png")
        with open(path, "wb") as f:
            f.write(data)
        return tile_id
    
    def load_tile(self, tile_id: str) -> Optional[bytes]:
        path = os.path.join(self.base_dir, f"{tile_id}.png")
        try:
            with open(path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None
    
    # Implement other methods...