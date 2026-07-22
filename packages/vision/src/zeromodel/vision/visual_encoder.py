"""Optional frozen visual encoders for Phase 1 address benchmarks.

The core package remains NumPy-only. Concrete learned encoders import their heavy
runtime dependencies lazily and are intended for explicit research runs, not for
implicit downloads during normal package import or CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

from zeromodel.core.artifact import VPMValidationError
from zeromodel.observation.visual_address import ImageObservation


ENCODER_MANIFEST_VERSION = "zeromodel-visual-encoder-manifest/v1"
DINO_V2_SMALL_MODEL_ID = "facebook/dinov2-small"
DINO_V2_SMALL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
DINO_V2_SMALL_LICENSE = "Apache-2.0"
LETTERBOX_CONTRACT_VERSION = "zeromodel-square-letterbox/v1"


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, np.generic):
        raise VPMValidationError("encoder manifest JSON must use plain scalar types")
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _thaw(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError(
            "encoder manifest values must be JSON-serializable"
        ) from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class EncoderManifest:
    """Identity record for one frozen visual representation contract."""

    provider_kind: str
    model_id: str
    revision: str
    architecture: str
    weights_digest: str
    preprocessing_digest: str
    output_dimension: int
    normalization: str
    framework: str
    framework_version: str
    license_id: str
    source_record: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ENCODER_MANIFEST_VERSION
    manifest_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if self.version != ENCODER_MANIFEST_VERSION:
            raise VPMValidationError("unsupported encoder manifest version")
        for name in (
            "provider_kind",
            "model_id",
            "revision",
            "architecture",
            "weights_digest",
            "preprocessing_digest",
            "normalization",
            "framework",
            "framework_version",
            "license_id",
            "source_record",
        ):
            if not str(getattr(self, name)):
                raise VPMValidationError("%s cannot be empty" % name)
        if self.output_dimension <= 0:
            raise VPMValidationError("encoder output_dimension must be positive")
        _json_bytes(self.metadata)
        expected = self.compute_manifest_id()
        if self.manifest_id is None:
            object.__setattr__(self, "manifest_id", expected)
        elif self.manifest_id != expected:
            raise VPMValidationError("encoder manifest id mismatch")

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "provider_kind": self.provider_kind,
            "model_id": self.model_id,
            "revision": self.revision,
            "architecture": self.architecture,
            "weights_digest": self.weights_digest,
            "preprocessing_digest": self.preprocessing_digest,
            "output_dimension": int(self.output_dimension),
            "normalization": self.normalization,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "license_id": self.license_id,
            "source_record": self.source_record,
            "metadata": _thaw(self.metadata),
        }

    def compute_manifest_id(self) -> str:
        return _sha256(
            b"zeromodel.visual-encoder-manifest.identity.v1\0"
            + _json_bytes(self.identity_payload())
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = self.identity_payload()
        payload["manifest_id"] = self.manifest_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EncoderManifest":
        return cls(**dict(data))


@runtime_checkable
class FrozenVisualEncoder(Protocol):
    """Batch encoder used by frozen representation benchmarks."""

    def manifest(self) -> EncoderManifest: ...

    def encode_batch(self, observations: Sequence[ImageObservation]) -> np.ndarray: ...


def square_letterbox_uint8(
    image: Any,
    *,
    canvas_side: int,
    fill: int = 0,
) -> np.ndarray:
    """Centre an RGB uint8 image on an owned square canvas without resizing."""

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise VPMValidationError("letterbox input must be HxWx3 uint8 RGB")
    if array.size == 0:
        raise VPMValidationError("letterbox input cannot be empty")
    if canvas_side < max(array.shape[0], array.shape[1]):
        raise VPMValidationError("letterbox canvas cannot crop the source image")
    if not (0 <= int(fill) <= 255):
        raise VPMValidationError("letterbox fill must be in [0, 255]")

    canvas = np.full(
        (int(canvas_side), int(canvas_side), 3),
        int(fill),
        dtype=np.uint8,
    )
    top = (int(canvas_side) - array.shape[0]) // 2
    left = (int(canvas_side) - array.shape[1]) // 2
    canvas[top : top + array.shape[0], left : left + array.shape[1]] = array
    canvas.flags.writeable = False
    return canvas


def _integer_processor_value(value: Any, *keys: str) -> Optional[int]:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, Mapping):
        for key in keys:
            item = value.get(key)
            if isinstance(item, (int, np.integer)):
                return int(item)
    for key in keys:
        item = getattr(value, key, None)
        if isinstance(item, (int, np.integer)):
            return int(item)
    return None


def _letterbox_canvas_side(processor: Any, height: int, width: int) -> int:
    """Choose a square canvas whose source survives configured centre cropping."""

    maximum = max(int(height), int(width))
    if maximum <= 0:
        raise VPMValidationError("letterbox source dimensions must be positive")
    do_crop = bool(getattr(processor, "do_center_crop", False))
    if not do_crop:
        return maximum

    processor_data = processor.to_dict() if hasattr(processor, "to_dict") else {}
    size_value = getattr(processor, "size", None)
    crop_value = getattr(processor, "crop_size", None)
    if size_value is None:
        size_value = processor_data.get("size")
    if crop_value is None:
        crop_value = processor_data.get("crop_size")
    resize_edge = _integer_processor_value(
        size_value,
        "shortest_edge",
        "height",
        "width",
    )
    crop_edge = _integer_processor_value(
        crop_value,
        "height",
        "width",
        "shortest_edge",
    )
    if resize_edge is None or crop_edge is None or resize_edge <= 0 or crop_edge <= 0:
        raise VPMValidationError(
            "centre-cropping processor must declare integer resize and crop sizes"
        )
    visible_fraction = min(1.0, float(crop_edge) / float(resize_edge))
    if visible_fraction <= 0.0:
        raise VPMValidationError("processor crop contract has no visible area")
    # Two source pixels of safety absorb integer resize/crop rounding at both sides.
    return max(maximum, int(math.ceil(float(maximum) / visible_fraction)) + 2)


def _canonical_processor_digest(
    processor: Any, canonicalization: Mapping[str, Any]
) -> str:
    if not hasattr(processor, "to_dict"):
        raise VPMValidationError("visual processor must expose to_dict()")
    payload = {
        "processor": processor.to_dict(),
        "canonicalization": dict(canonicalization),
    }
    return _sha256(b"zeromodel.visual-preprocessing.v2\0" + _json_bytes(payload))


def _torch_state_dict_digest(torch_module: Any, state_dict: Mapping[str, Any]) -> str:
    """Hash sorted tensor names, dtypes, shapes, and raw bytes.

    This is deliberately computed from the loaded state dictionary rather than
    trusting a mutable model alias. It is expensive but performed once when the
    research encoder is constructed.
    """

    digest = hashlib.sha256(b"zeromodel.torch-state-dict.v1\0")
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(len(name.encode("utf-8")).to_bytes(4, "big"))
        digest.update(name.encode("utf-8"))
        dtype_name = str(tensor.dtype)
        digest.update(len(dtype_name.encode("utf-8")).to_bytes(4, "big"))
        digest.update(dtype_name.encode("utf-8"))
        digest.update(_json_bytes(list(tensor.shape)))
        byte_view = tensor.view(torch_module.uint8)
        digest.update(byte_view.numpy().tobytes(order="C"))
    return digest.hexdigest()


class HuggingFaceDinoV2Encoder:
    """Pinned DINOv2 global embedding adapter with crop-safe letterboxing.

    Heavy dependencies are loaded lazily. Construction may download model files
    unless ``local_files_only=True``. Runtime output is the L2-normalized CLS
    token from ``last_hidden_state`` as an immutable float32 matrix.

    Wide or tall observations are first centred on a square canvas large enough
    that the processor's declared centre crop cannot discard source pixels. The
    letterbox operation and processor configuration are part of manifest identity.
    """

    def __init__(
        self,
        *,
        model_id: str = DINO_V2_SMALL_MODEL_ID,
        revision: str = DINO_V2_SMALL_REVISION,
        device: str = "cpu",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        letterbox_fill: int = 0,
    ) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModel
            import transformers
        except ImportError as exc:
            raise VPMValidationError(
                "HuggingFaceDinoV2Encoder requires the optional 'vision' dependencies"
            ) from exc

        if not (0 <= int(letterbox_fill) <= 255):
            raise VPMValidationError("letterbox_fill must be in [0, 255]")
        self._torch = torch
        self._image_type = Image
        self._device = str(device)
        self._letterbox_fill = int(letterbox_fill)
        self._processor = AutoImageProcessor.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self._model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self._model.eval()
        self._model.to(self._device)

        output_dimension = int(getattr(self._model.config, "hidden_size", 0))
        if output_dimension <= 0:
            raise VPMValidationError("DINOv2 model config must declare hidden_size")

        canonicalization = {
            "version": LETTERBOX_CONTRACT_VERSION,
            "kind": "square_letterbox_with_center_crop_guard",
            "fill": self._letterbox_fill,
            "source_resize": "none",
            "placement": "integer_center",
            "safety_pixels": 2,
        }
        weights_digest = _torch_state_dict_digest(torch, self._model.state_dict())
        preprocessing_digest = _canonical_processor_digest(
            self._processor,
            canonicalization,
        )
        self._manifest = EncoderManifest(
            provider_kind="huggingface_dinov2_cls",
            model_id=str(model_id),
            revision=str(revision),
            architecture=str(getattr(self._model.config, "model_type", "dinov2")),
            weights_digest=weights_digest,
            preprocessing_digest=preprocessing_digest,
            output_dimension=output_dimension,
            normalization="l2",
            framework="transformers+pytorch",
            framework_version="transformers=%s;torch=%s"
            % (str(transformers.__version__), str(torch.__version__)),
            license_id=DINO_V2_SMALL_LICENSE,
            source_record="https://huggingface.co/%s/tree/%s" % (model_id, revision),
            metadata={
                "feature": "last_hidden_state[:,0,:]",
                "device": self._device,
                "trust_remote_code": bool(trust_remote_code),
                "canonicalization": canonicalization,
            },
        )

    def manifest(self) -> EncoderManifest:
        return self._manifest

    def _to_pil(self, observation: ImageObservation) -> Any:
        array = observation.pixels
        if array.ndim == 2:
            rgb = np.repeat(array[:, :, None], 3, axis=2)
        elif array.shape[2] == 4:
            rgb = array[:, :, :3]
        else:
            rgb = array
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        side = _letterbox_canvas_side(
            self._processor,
            rgb.shape[0],
            rgb.shape[1],
        )
        letterboxed = square_letterbox_uint8(
            rgb,
            canvas_side=side,
            fill=self._letterbox_fill,
        )
        return self._image_type.fromarray(letterboxed)

    def encode_batch(self, observations: Sequence[ImageObservation]) -> np.ndarray:
        items: Tuple[ImageObservation, ...] = tuple(observations)
        if not items:
            raise VPMValidationError("frozen encoder batch cannot be empty")
        if any(not isinstance(item, ImageObservation) for item in items):
            raise VPMValidationError("frozen encoder requires ImageObservation values")

        images = [self._to_pil(item) for item in items]
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._model(**inputs)
            if not hasattr(output, "last_hidden_state"):
                raise VPMValidationError("DINOv2 output requires last_hidden_state")
            vectors = output.last_hidden_state[:, 0, :]
            vectors = self._torch.nn.functional.normalize(vectors, p=2.0, dim=1)
        array = np.ascontiguousarray(vectors.detach().cpu().numpy(), dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self._manifest.output_dimension:
            raise VPMValidationError("frozen encoder output shape violates manifest")
        if not np.isfinite(array).all():
            raise VPMValidationError("frozen encoder output must be finite")
        array.flags.writeable = False
        return array
