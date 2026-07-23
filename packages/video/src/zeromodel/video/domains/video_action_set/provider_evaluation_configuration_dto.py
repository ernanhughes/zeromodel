"""`ProviderConfigurationDTO` - the provider/model/runtime identity for one run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    PROVIDER_EVALUATION_CONFIGURATION_VERSION,
)
from zeromodel.video.domains.video_action_set.dto import CanonicalJsonDTO
from zeromodel.video.domains.video_action_set.observation_common import (
    json_mapping,
    optional_string,
    require_keys,
    sha256,
    string,
)
from zeromodel.video.domains.video_action_set.provider_evaluation_common import (
    nonempty_str,
    nonneg_int,
    optional_nonempty_str,
    optional_nonneg_int,
)


_FORBIDDEN_CONFIG_KEY_FRAGMENTS = (
    "apikey",
    "token",
    "password",
    "secret",
    "bearer",
    "authorization",
    "credential",
)

CONFIGURATION_KEYS = (
    "version",
    "provider_kind",
    "provider_version",
    "model_name",
    "model_digest",
    "runtime_name",
    "runtime_version",
    "protocol_version",
    "prompt_digest",
    "context_length",
    "seed",
    "inference_options",
    "metadata",
    "provider_configuration_id",
)


def _normalize_key(key: str) -> str:
    return "".join(character for character in key.lower() if character.isalnum())


def _reject_secret_keys(value: object, message: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalize_key(str(key))
            if any(
                fragment in normalized for fragment in _FORBIDDEN_CONFIG_KEY_FRAGMENTS
            ):
                raise VPMValidationError(message)
            _reject_secret_keys(item, message)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_secret_keys(item, message)


@dataclass(frozen=True, slots=True)
class ProviderConfigurationDTO:
    """The provider/model/runtime configuration used for one evaluation run.

    Identity is content-derived: no wall-clock field participates in
    ``provider_configuration_id``. Secrets, API keys, bearer tokens and
    credentials must never appear in ``inference_options``/``metadata`` -
    enforced by a recursive key-fragment check, not merely documented.
    """

    version: str
    provider_kind: str
    provider_version: str | None
    model_name: str
    model_digest: str
    runtime_name: str
    runtime_version: str | None
    protocol_version: str
    prompt_digest: str
    context_length: int | None
    seed: int | None
    inference_options: CanonicalJsonDTO
    metadata: CanonicalJsonDTO
    provider_configuration_id: str

    def __post_init__(self) -> None:
        if self.version != PROVIDER_EVALUATION_CONFIGURATION_VERSION:
            raise VPMValidationError("unsupported provider configuration version")
        nonempty_str(self.provider_kind, "provider configuration kind mismatch")
        optional_nonempty_str(
            self.provider_version, "provider configuration version mismatch"
        )
        nonempty_str(self.model_name, "provider configuration model name mismatch")
        sha256(self.model_digest, "provider configuration model digest mismatch")
        nonempty_str(self.runtime_name, "provider configuration runtime name mismatch")
        optional_nonempty_str(
            self.runtime_version, "provider configuration runtime version mismatch"
        )
        nonempty_str(
            self.protocol_version, "provider configuration protocol version mismatch"
        )
        sha256(self.prompt_digest, "provider configuration prompt digest mismatch")
        if self.context_length is not None:
            nonneg_int(
                self.context_length,
                "provider configuration context length mismatch",
            )
        if self.seed is not None:
            nonneg_int(self.seed, "provider configuration seed mismatch")
        options = json_mapping(
            self.inference_options,
            "provider configuration inference options mismatch",
        )
        _reject_secret_keys(
            options,
            "provider configuration inference options must not contain secrets",
        )
        metadata = json_mapping(
            self.metadata, "provider configuration metadata mismatch"
        )
        _reject_secret_keys(
            metadata, "provider configuration metadata must not contain secrets"
        )
        expected_id = canonical_sha256(_configuration_payload_without_id(self))
        if self.provider_configuration_id != expected_id:
            raise VPMValidationError("provider configuration id mismatch")

    @classmethod
    def build(
        cls,
        *,
        provider_kind: str,
        model_name: str,
        model_digest: str,
        runtime_name: str,
        protocol_version: str,
        prompt_digest: str,
        seed: int | None = None,
        inference_options: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "ProviderConfigurationDTO":
        """Ergonomic builder for the common case.

        ``provider_version``/``runtime_version``/``context_length`` are
        rarely-used identity fields with no current caller; set them via
        ``from_dict`` directly if a future caller needs them, rather than
        growing this constructor's parameter count speculatively.
        """
        payload = {
            "version": PROVIDER_EVALUATION_CONFIGURATION_VERSION,
            "provider_kind": provider_kind,
            "provider_version": None,
            "model_name": model_name,
            "model_digest": model_digest,
            "runtime_name": runtime_name,
            "runtime_version": None,
            "protocol_version": protocol_version,
            "prompt_digest": prompt_digest,
            "context_length": None,
            "seed": seed,
            "inference_options": dict(inference_options or {}),
            "metadata": dict(metadata or {}),
        }
        configuration_id = canonical_sha256(payload)
        return cls.from_dict(payload | {"provider_configuration_id": configuration_id})

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProviderConfigurationDTO":
        require_keys(
            payload, CONFIGURATION_KEYS, "provider configuration keys mismatch"
        )
        return cls(
            version=string(
                payload, "version", "unsupported provider configuration version"
            ),
            provider_kind=string(
                payload, "provider_kind", "provider configuration kind mismatch"
            ),
            provider_version=optional_string(
                payload,
                "provider_version",
                "provider configuration version mismatch",
            ),
            model_name=string(
                payload, "model_name", "provider configuration model name mismatch"
            ),
            model_digest=sha256(
                payload["model_digest"],
                "provider configuration model digest mismatch",
            ),
            runtime_name=string(
                payload,
                "runtime_name",
                "provider configuration runtime name mismatch",
            ),
            runtime_version=optional_string(
                payload,
                "runtime_version",
                "provider configuration runtime version mismatch",
            ),
            protocol_version=string(
                payload,
                "protocol_version",
                "provider configuration protocol version mismatch",
            ),
            prompt_digest=sha256(
                payload["prompt_digest"],
                "provider configuration prompt digest mismatch",
            ),
            context_length=optional_nonneg_int(
                payload.get("context_length"),
                "provider configuration context length mismatch",
            ),
            seed=optional_nonneg_int(
                payload.get("seed"), "provider configuration seed mismatch"
            ),
            inference_options=CanonicalJsonDTO.from_value(payload["inference_options"]),
            metadata=CanonicalJsonDTO.from_value(payload["metadata"]),
            provider_configuration_id=string(
                payload,
                "provider_configuration_id",
                "provider configuration id mismatch",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "provider_kind": self.provider_kind,
            "provider_version": self.provider_version,
            "model_name": self.model_name,
            "model_digest": self.model_digest,
            "runtime_name": self.runtime_name,
            "runtime_version": self.runtime_version,
            "protocol_version": self.protocol_version,
            "prompt_digest": self.prompt_digest,
            "context_length": self.context_length,
            "seed": self.seed,
            "inference_options": self.inference_options.to_value(),
            "metadata": self.metadata.to_value(),
            "provider_configuration_id": self.provider_configuration_id,
        }


def _configuration_payload_without_id(
    configuration: ProviderConfigurationDTO,
) -> dict[str, object]:
    payload = configuration.to_dict()
    payload.pop("provider_configuration_id")
    return payload


__all__ = ["CONFIGURATION_KEYS", "ProviderConfigurationDTO"]
