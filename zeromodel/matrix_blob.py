"""Content-addressed canonical matrix storage for non-VPM tensors.

``MatrixBlob`` exists for dense representations whose columns are not human-
meaningful VPM metrics: embeddings, patch banks, projections, or other tensor
payloads. It gives those bytes a deterministic identity without pretending
that each vector dimension is an inspectable policy metric.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifact import VPMValidationError


MATRIX_BLOB_VERSION = "zeromodel-matrix-blob/v1"

_DTYPE_SPECS = {
    "int8": (np.dtype("int8"), np.dtype("int8")),
    "uint8": (np.dtype("uint8"), np.dtype("uint8")),
    "int16": (np.dtype("int16"), np.dtype(">i2")),
    "int32": (np.dtype("int32"), np.dtype(">i4")),
    "float32": (np.dtype("float32"), np.dtype(">f4")),
    "float64": (np.dtype("float64"), np.dtype(">f8")),
}


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        raise VPMValidationError("matrix blob metadata must use plain JSON scalar types")
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _thaw_json(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError(
            "matrix blob metadata must be JSON-serializable"
        ) from exc


def _length_prefixed(label: str, data: bytes) -> bytes:
    label_bytes = label.encode("utf-8")
    return (
        len(label_bytes).to_bytes(4, "big")
        + label_bytes
        + len(data).to_bytes(8, "big")
        + data
    )


def _canonical_dtype_name(dtype: np.dtype[Any]) -> str:
    native = np.dtype(dtype).newbyteorder("=")
    for name, (expected_native, _) in _DTYPE_SPECS.items():
        if native == expected_native:
            return name
    raise VPMValidationError(
        "matrix blob dtype must be one of: %s"
        % ", ".join(sorted(_DTYPE_SPECS))
    )


def _canonical_array_bytes(array: np.ndarray, dtype_name: str) -> bytes:
    _, storage_dtype = _DTYPE_SPECS[dtype_name]
    canonical = np.ascontiguousarray(array.astype(storage_dtype, copy=False))
    return canonical.tobytes(order="C")


@dataclass(frozen=True)
class MatrixBlob:
    """Immutable content-addressed tensor payload.

    ``scale`` and ``zero_point`` are identity-bearing quantization metadata. They
    describe how a consumer may interpret integer values; they do not alter the
    canonical bytes stored by this object.
    """

    dtype: str
    shape: Tuple[int, ...]
    data: bytes
    scale: Optional[float] = None
    zero_point: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = MATRIX_BLOB_VERSION
    blob_id: str = ""

    def __init__(
        self,
        *,
        dtype: str,
        shape: Sequence[int],
        data: bytes,
        scale: Optional[float] = None,
        zero_point: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        version: str = MATRIX_BLOB_VERSION,
        blob_id: Optional[str] = None,
    ) -> None:
        object.__setattr__(self, "dtype", str(dtype))
        object.__setattr__(self, "shape", tuple(int(value) for value in shape))
        object.__setattr__(self, "data", bytes(data))
        object.__setattr__(self, "scale", None if scale is None else float(scale))
        object.__setattr__(
            self,
            "zero_point",
            None if zero_point is None else int(zero_point),
        )
        object.__setattr__(self, "metadata", _freeze_json(metadata or {}))
        object.__setattr__(self, "version", str(version))
        computed = self.compute_blob_id()
        object.__setattr__(self, "blob_id", str(blob_id or computed))
        self.validate()

    @classmethod
    def from_array(
        cls,
        values: Any,
        *,
        dtype: Optional[str] = None,
        scale: Optional[float] = None,
        zero_point: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "MatrixBlob":
        array = np.asarray(values)
        if array.ndim < 1:
            raise VPMValidationError("matrix blob requires at least one dimension")
        dtype_name = str(dtype) if dtype is not None else _canonical_dtype_name(array.dtype)
        if dtype_name not in _DTYPE_SPECS:
            raise VPMValidationError("unsupported matrix blob dtype: %r" % dtype_name)
        native_dtype, _ = _DTYPE_SPECS[dtype_name]
        converted = np.asarray(array, dtype=native_dtype)
        if np.issubdtype(native_dtype, np.floating) and not np.isfinite(converted).all():
            raise VPMValidationError("matrix blob floating values must be finite")
        return cls(
            dtype=dtype_name,
            shape=converted.shape,
            data=_canonical_array_bytes(converted, dtype_name),
            scale=scale,
            zero_point=zero_point,
            metadata=metadata,
        )

    def validate(self) -> None:
        if self.version != MATRIX_BLOB_VERSION:
            raise VPMValidationError(
                "unsupported matrix blob version: %r" % self.version
            )
        if self.dtype not in _DTYPE_SPECS:
            raise VPMValidationError("unsupported matrix blob dtype: %r" % self.dtype)
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise VPMValidationError(
                "matrix blob shape must contain positive dimensions"
            )
        native_dtype, _ = _DTYPE_SPECS[self.dtype]
        expected_size = int(np.prod(self.shape, dtype=np.int64)) * native_dtype.itemsize
        if len(self.data) != expected_size:
            raise VPMValidationError(
                "matrix blob byte length mismatch: expected %d, got %d"
                % (expected_size, len(self.data))
            )
        if self.scale is not None:
            if not np.isfinite(self.scale) or self.scale <= 0.0:
                raise VPMValidationError("matrix blob scale must be positive and finite")
            if not np.issubdtype(native_dtype, np.integer):
                raise VPMValidationError(
                    "matrix blob scale is only valid for integer storage"
                )
        if self.zero_point is not None:
            if not np.issubdtype(native_dtype, np.integer):
                raise VPMValidationError(
                    "matrix blob zero_point is only valid for integer storage"
                )
            info = np.iinfo(native_dtype)
            if not (int(info.min) <= self.zero_point <= int(info.max)):
                raise VPMValidationError(
                    "matrix blob zero_point is outside the storage dtype range"
                )
        _canonical_json_bytes(self.metadata)
        expected_id = self.compute_blob_id()
        if self.blob_id != expected_id:
            raise VPMValidationError(
                "matrix blob id mismatch: expected %s, got %s"
                % (expected_id, self.blob_id)
            )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def nbytes(self) -> int:
        return len(self.data)

    def identity_bytes(self) -> bytes:
        quantization = {"scale": self.scale, "zero_point": self.zero_point}
        return b"".join(
            _length_prefixed(label, value)
            for label, value in (
                ("format", b"zeromodel.matrix-blob.identity.v1"),
                ("version", self.version.encode("utf-8")),
                ("dtype", self.dtype.encode("utf-8")),
                ("shape", _canonical_json_bytes(list(self.shape))),
                ("quantization", _canonical_json_bytes(quantization)),
                ("metadata", _canonical_json_bytes(self.metadata)),
                ("data", self.data),
            )
        )

    def compute_blob_id(self) -> str:
        return hashlib.sha256(self.identity_bytes()).hexdigest()

    def to_array(self, *, dequantize: bool = False) -> np.ndarray:
        native_dtype, storage_dtype = _DTYPE_SPECS[self.dtype]
        array = np.frombuffer(self.data, dtype=storage_dtype).astype(
            native_dtype,
            copy=True,
        )
        array = np.ascontiguousarray(array.reshape(self.shape))
        if dequantize:
            if self.scale is None:
                raise VPMValidationError(
                    "matrix blob has no scale for dequantization"
                )
            zero_point = 0 if self.zero_point is None else self.zero_point
            array = (array.astype(np.float32) - float(zero_point)) * float(self.scale)
        array.flags.writeable = False
        return array

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "blob_id": self.blob_id,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "scale": self.scale,
            "zero_point": self.zero_point,
            "metadata": _thaw_json(self.metadata),
            "data_base64": base64.b64encode(self.data).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MatrixBlob":
        try:
            raw = base64.b64decode(str(data["data_base64"]), validate=True)
        except (KeyError, ValueError) as exc:
            raise VPMValidationError("invalid matrix blob base64 payload") from exc
        return cls(
            dtype=str(data["dtype"]),
            shape=data["shape"],
            data=raw,
            scale=data.get("scale"),
            zero_point=data.get("zero_point"),
            metadata=data.get("metadata") or {},
            version=str(data.get("version", MATRIX_BLOB_VERSION)),
            blob_id=data.get("blob_id"),
        )
