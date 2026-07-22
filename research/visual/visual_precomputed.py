"""Provider adapter for benchmark representations extracted exactly once."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from zeromodel.core.artifact import VPMValidationError
from zeromodel.observation.visual_address import (
    ImageObservation,
    VisualAddressContract,
    VisualAddressDecision,
)


@runtime_checkable
class VectorAddressMatcher(Protocol):
    def contract(self) -> VisualAddressContract: ...

    def match_vector(
        self,
        vector: object,
        *,
        observation_digest: str,
    ) -> VisualAddressDecision: ...


class PrecomputedVectorAddressProvider:
    """Serve one immutable representation per canonical observation digest.

    The adapter is intended for controlled benchmarks where a pinned encoder has
    already extracted every declared observation exactly once. Multiple matchers
    can then consume the same vectors without repeating model inference.
    """

    def __init__(
        self,
        matcher: VectorAddressMatcher,
        vectors_by_observation_digest: Mapping[str, object],
    ) -> None:
        if not isinstance(matcher, VectorAddressMatcher):
            raise VPMValidationError("precomputed provider requires a vector matcher")
        vectors = {}
        dimensions = set()
        for digest, value in vectors_by_observation_digest.items():
            key = str(digest)
            if not key:
                raise VPMValidationError(
                    "precomputed observation digest cannot be empty"
                )
            vector = np.asarray(value, dtype=np.float32).reshape(-1)
            if vector.size == 0 or not np.isfinite(vector).all():
                raise VPMValidationError(
                    "precomputed representations must be non-empty and finite"
                )
            owned = np.ascontiguousarray(vector, dtype=np.float32)
            owned.flags.writeable = False
            vectors[key] = owned
            dimensions.add(int(owned.size))
        if not vectors:
            raise VPMValidationError("precomputed provider requires representations")
        if len(dimensions) != 1:
            raise VPMValidationError(
                "precomputed representations must share one dimension"
            )
        self.matcher = matcher
        self._vectors = MappingProxyType(vectors)
        self.representation_dimension = next(iter(dimensions))

    def contract(self) -> VisualAddressContract:
        return self.matcher.contract()

    def read(self, observation: ImageObservation) -> VisualAddressDecision:
        if not isinstance(observation, ImageObservation):
            raise VPMValidationError("precomputed provider requires ImageObservation")
        try:
            vector = self._vectors[observation.raw_digest]
        except KeyError as exc:
            raise VPMValidationError(
                "observation has no precomputed representation: %s"
                % observation.raw_digest
            ) from exc
        decision = self.matcher.match_vector(
            vector,
            observation_digest=observation.raw_digest,
        )
        if decision.observation_digest != observation.raw_digest:
            raise VPMValidationError(
                "vector matcher returned the wrong observation digest"
            )
        return decision
