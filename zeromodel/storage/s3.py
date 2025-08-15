from zeromodel.storage.base import StorageBackend
from typing import List, Tuple, Optional
import boto3
import logging

logger = logging.getLogger(__name__)


class S3Storage(StorageBackend[bytes]):
    """S3 storage backend for production world-scale deployments."""

    def __init__(self, bucket_name: str, prefix: str = "vpm/"):
        try:
            self.s3 = boto3.client("s3")
            self.bucket = bucket_name
            self.prefix = prefix
        except ImportError:
            logger.error(
                "boto3 is required for S3Storage. Install with 'pip install boto3'"
            )
            raise

    def store_tile(self, level: int, x: int, y: int, data: bytes) -> str:
        tile_id = self.get_tile_id(level, x, y)
        key = f"{self.prefix}{tile_id}.png"
        self.s3.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType="image/png"
        )
        return tile_id

    def load_tile(self, tile_id: str) -> Optional[bytes]:
        key = f"{self.prefix}{tile_id}.png"
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def query_region(
        self, level: int, x_start: int, y_start: int, x_end: int, y_end: int
    ) -> List[Tuple[int, int, bytes]]:
        results = []
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                tile_id = self.get_tile_id(level, x, y)
                tile = self.load_tile(tile_id)
                if tile is not None:
                    results.append((x, y, tile))
        return results

    def create_index(self, level: int, index_type: str = "spatial") -> None:
        # In a production system, you might create a DynamoDB index
        # or use S3 metadata for spatial queries
        pass
