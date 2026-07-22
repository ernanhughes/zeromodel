"""DTOs for the finite, identified artifact-corpus hierarchy.

Every content-derived identity field (`tile_id`, `leaf_id`, `hierarchy_id`,
`receipt_id`) follows the same self-validating pattern already used by
`zeromodel.core.artifact.VPMArtifact.artifact_id` and
`zeromodel.trust.dto.ArtifactAuthorizationDTO.authorization_id`: the field
is recomputed from the object's own canonical content in `__post_init__`
and construction fails if it does not match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from zeromodel.artifacts import (
    ArtifactRef,
    canonical_json_bytes,
    is_sha256_digest,
    sha256_digest,
)
from zeromodel.core.artifact import VPMValidationError

SPEC_VERSION = "zeromodel-navigation/v1"


def _require_nonempty_str(value: object, message: str) -> None:
    if not isinstance(value, str) or not value:
        raise VPMValidationError(message)


def _pairs_to_dict(pairs: Tuple[Tuple[str, str], ...], context: str) -> dict:
    """Convert declared key/value pairs to a dict, rejecting duplicate
    keys rather than silently keeping the last one (plain `dict(pairs)`
    would)."""
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise VPMValidationError(f"{context} has a duplicate key: {key!r}")
        result[key] = value
    return result


def _require_sha256(value: str, message: str) -> None:
    if not is_sha256_digest(value):
        raise VPMValidationError(message)


@dataclass(frozen=True, slots=True)
class TileCoverageDTO:
    """What corpus partition a tile represents - used by closure checks."""

    corpus_id: str
    partition_key: str
    child_count: int
    leaf_count: int
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.corpus_id, "TileCoverageDTO.corpus_id must be non-empty"
        )
        _require_nonempty_str(
            self.partition_key, "TileCoverageDTO.partition_key must be non-empty"
        )
        if self.child_count < 0 or self.leaf_count < 0:
            raise VPMValidationError("TileCoverageDTO counts must be >= 0")


@dataclass(frozen=True, slots=True)
class TilePointerDTO:
    """A reference to one child - another tile, or a leaf binding.

    `order_key` is an explicit, declared ordering key: child order is part
    of a tile's identity, never left to incidental container ordering.
    """

    pointer_kind: str  # "tile" | "leaf"
    target_id: str  # sha256:... digest of the referenced tile/leaf binding
    order_key: str
    spec_version: str = SPEC_VERSION

    _VALID_POINTER_KINDS = ("tile", "leaf")

    def __post_init__(self) -> None:
        if self.pointer_kind not in self._VALID_POINTER_KINDS:
            raise VPMValidationError(
                f"TilePointerDTO.pointer_kind must be one of {self._VALID_POINTER_KINDS}"
            )
        _require_sha256(
            self.target_id, "TilePointerDTO.target_id must be a sha256: digest"
        )
        _require_nonempty_str(
            self.order_key, "TilePointerDTO.order_key must be non-empty"
        )

    def as_payload(self) -> dict:
        return {
            "pointer_kind": self.pointer_kind,
            "target_id": self.target_id,
            "order_key": self.order_key,
        }


@dataclass(frozen=True, slots=True)
class LeafBindingDTO:
    """A leaf: binds one navigation position to exactly one identified artifact."""

    leaf_id: str
    artifact_ref: ArtifactRef
    leaf_semantics: str
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_sha256(self.leaf_id, "LeafBindingDTO.leaf_id must be a sha256: digest")
        _require_nonempty_str(
            self.leaf_semantics, "LeafBindingDTO.leaf_semantics must be non-empty"
        )
        expected_id = sha256_digest(
            canonical_json_bytes(leaf_binding_identity_payload(self))
        )
        if self.leaf_id != expected_id:
            raise VPMValidationError(
                "LeafBindingDTO.leaf_id does not match its own canonical content"
            )


def leaf_binding_identity_payload(binding: LeafBindingDTO) -> dict:
    return {
        "spec_version": binding.spec_version,
        "artifact_kind": binding.artifact_ref.artifact_kind,
        "artifact_id": binding.artifact_ref.artifact_id,
        "leaf_semantics": binding.leaf_semantics,
    }


def compute_leaf_id(
    *, artifact_ref: ArtifactRef, leaf_semantics: str, spec_version: str = SPEC_VERSION
) -> str:
    payload = {
        "spec_version": spec_version,
        "artifact_kind": artifact_ref.artifact_kind,
        "artifact_id": artifact_ref.artifact_id,
        "leaf_semantics": leaf_semantics,
    }
    return sha256_digest(canonical_json_bytes(payload))


@dataclass(frozen=True, slots=True)
class NavigationTileDTO:
    """One node (root or internal) in the artifact-corpus hierarchy.

    Contains no live callbacks - only stable child pointers and a coverage
    descriptor. `tile_id` binds depth, coverage, and the exact ordered
    sequence of children: reordering children changes this tile's identity.
    """

    tile_id: str
    depth: int
    coverage: TileCoverageDTO
    children: Tuple[TilePointerDTO, ...]
    tie_rule: str
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.tile_id, "NavigationTileDTO.tile_id must be a sha256: digest"
        )
        if self.depth < 0:
            raise VPMValidationError("NavigationTileDTO.depth must be >= 0")
        if not self.children:
            raise VPMValidationError("NavigationTileDTO.children must not be empty")
        _require_nonempty_str(
            self.tie_rule, "NavigationTileDTO.tie_rule must be non-empty"
        )
        seen_targets = set()
        seen_order_keys = set()
        for child in self.children:
            if child.target_id in seen_targets:
                raise VPMValidationError(
                    f"NavigationTileDTO has a duplicate child target: {child.target_id}"
                )
            if child.order_key in seen_order_keys:
                raise VPMValidationError(
                    f"NavigationTileDTO has a duplicate child order_key: {child.order_key}"
                )
            seen_targets.add(child.target_id)
            seen_order_keys.add(child.order_key)
        expected_id = sha256_digest(canonical_json_bytes(tile_identity_payload(self)))
        if self.tile_id != expected_id:
            raise VPMValidationError(
                "NavigationTileDTO.tile_id does not match its own canonical content"
            )


def tile_identity_payload(tile: NavigationTileDTO) -> dict:
    return {
        "spec_version": tile.spec_version,
        "depth": tile.depth,
        "coverage": {
            "corpus_id": tile.coverage.corpus_id,
            "partition_key": tile.coverage.partition_key,
            "child_count": tile.coverage.child_count,
            "leaf_count": tile.coverage.leaf_count,
        },
        "children": [child.as_payload() for child in tile.children],
        "tie_rule": tile.tie_rule,
    }


def compute_tile_id(
    *,
    depth: int,
    coverage: TileCoverageDTO,
    children: Tuple[TilePointerDTO, ...],
    tie_rule: str,
    spec_version: str = SPEC_VERSION,
) -> str:
    payload = {
        "spec_version": spec_version,
        "depth": depth,
        "coverage": {
            "corpus_id": coverage.corpus_id,
            "partition_key": coverage.partition_key,
            "child_count": coverage.child_count,
            "leaf_count": coverage.leaf_count,
        },
        "children": [child.as_payload() for child in children],
        "tie_rule": tie_rule,
    }
    return sha256_digest(canonical_json_bytes(payload))


@dataclass(frozen=True, slots=True)
class HierarchyCompilerSpecDTO:
    """Identifies the compiler and every parameter that determines a
    compiled hierarchy's structure, deterministically."""

    compiler_id: str
    compiler_version: str
    corpus_id: str
    corpus_artifact_kind: str
    leaf_semantics: str
    max_children_per_tile: int
    max_depth: int
    tie_rule: str
    failure_rule: str
    navigation_rule_contract: str
    child_ordering_rule: str = "declared-input-order"
    partition_parameters: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.compiler_id, "HierarchyCompilerSpecDTO.compiler_id must be non-empty"
        )
        _require_nonempty_str(
            self.compiler_version,
            "HierarchyCompilerSpecDTO.compiler_version must be non-empty",
        )
        _require_nonempty_str(
            self.corpus_id, "HierarchyCompilerSpecDTO.corpus_id must be non-empty"
        )
        _require_nonempty_str(
            self.corpus_artifact_kind,
            "HierarchyCompilerSpecDTO.corpus_artifact_kind must be non-empty",
        )
        _require_nonempty_str(
            self.leaf_semantics,
            "HierarchyCompilerSpecDTO.leaf_semantics must be non-empty",
        )
        if self.max_children_per_tile < 2:
            raise VPMValidationError(
                "HierarchyCompilerSpecDTO.max_children_per_tile must be >= 2"
            )
        if self.max_depth < 1:
            raise VPMValidationError("HierarchyCompilerSpecDTO.max_depth must be >= 1")


@dataclass(frozen=True, slots=True)
class HierarchyManifestDTO:
    """The identity of one compiled hierarchy.

    Binds the root tile reference, the source artifact set, compiler
    identity/version, partition parameters, child ordering, tie rule,
    failure rule, leaf semantics, and the navigation rule contract - never
    just a root pointer alone.
    """

    hierarchy_id: str
    root_ref: ArtifactRef
    source_artifact_digest: str
    compiler_id: str
    compiler_version: str
    corpus_id: str
    corpus_artifact_kind: str
    leaf_semantics: str
    max_children_per_tile: int
    max_depth: int
    tie_rule: str
    failure_rule: str
    navigation_rule_contract: str
    child_ordering_rule: str
    partition_parameters: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.hierarchy_id,
            "HierarchyManifestDTO.hierarchy_id must be a sha256: digest",
        )
        _require_sha256(
            self.source_artifact_digest,
            "HierarchyManifestDTO.source_artifact_digest must be a sha256: digest",
        )
        expected_id = sha256_digest(
            canonical_json_bytes(hierarchy_identity_payload(self))
        )
        if self.hierarchy_id != expected_id:
            raise VPMValidationError(
                "HierarchyManifestDTO.hierarchy_id does not match its own canonical content"
            )


def hierarchy_identity_payload(manifest: HierarchyManifestDTO) -> dict:
    return {
        "spec_version": manifest.spec_version,
        "root_artifact_kind": manifest.root_ref.artifact_kind,
        "root_artifact_id": manifest.root_ref.artifact_id,
        "source_artifact_digest": manifest.source_artifact_digest,
        "compiler_id": manifest.compiler_id,
        "compiler_version": manifest.compiler_version,
        "corpus_id": manifest.corpus_id,
        "corpus_artifact_kind": manifest.corpus_artifact_kind,
        "leaf_semantics": manifest.leaf_semantics,
        "max_children_per_tile": manifest.max_children_per_tile,
        "max_depth": manifest.max_depth,
        "tie_rule": manifest.tie_rule,
        "failure_rule": manifest.failure_rule,
        "navigation_rule_contract": manifest.navigation_rule_contract,
        "child_ordering_rule": manifest.child_ordering_rule,
        "partition_parameters": _pairs_to_dict(
            manifest.partition_parameters, "HierarchyManifestDTO.partition_parameters"
        ),
    }


def compute_hierarchy_id(
    *,
    root_ref: ArtifactRef,
    source_artifact_digest: str,
    spec: HierarchyCompilerSpecDTO,
) -> str:
    payload = {
        "spec_version": spec.spec_version,
        "root_artifact_kind": root_ref.artifact_kind,
        "root_artifact_id": root_ref.artifact_id,
        "source_artifact_digest": source_artifact_digest,
        "compiler_id": spec.compiler_id,
        "compiler_version": spec.compiler_version,
        "corpus_id": spec.corpus_id,
        "corpus_artifact_kind": spec.corpus_artifact_kind,
        "leaf_semantics": spec.leaf_semantics,
        "max_children_per_tile": spec.max_children_per_tile,
        "max_depth": spec.max_depth,
        "tie_rule": spec.tie_rule,
        "failure_rule": spec.failure_rule,
        "navigation_rule_contract": spec.navigation_rule_contract,
        "child_ordering_rule": spec.child_ordering_rule,
        "partition_parameters": _pairs_to_dict(
            spec.partition_parameters, "HierarchyCompilerSpecDTO.partition_parameters"
        ),
    }
    return sha256_digest(canonical_json_bytes(payload))


def compute_source_artifact_digest(artifact_refs: Tuple[ArtifactRef, ...]) -> str:
    """Order-independent identity of the source artifact set."""
    ids = sorted(ref.artifact_id for ref in artifact_refs)
    return sha256_digest({"artifact_ids": ids})


@dataclass(frozen=True, slots=True)
class TraversalRuleDescriptorDTO:
    """A stable, data-only identity for a traversal rule implementation.

    Never a live callback - descriptors are compared/recorded, and a
    production `TraversalRule` implementation is looked up or constructed
    from one out-of-band.
    """

    rule_kind: str
    parameters: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.rule_kind, "TraversalRuleDescriptorDTO.rule_kind must be non-empty"
        )


@dataclass(frozen=True, slots=True)
class TraversalRequestDTO:
    """A declared routing request - not a similarity query.

    `attributes` are explicit key/value routing criteria (e.g. a target
    range key, a fixed selector value) that a `TraversalRule` consults.
    """

    request_id: str
    attributes: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.request_id, "TraversalRequestDTO.request_id must be non-empty"
        )

    @property
    def attributes_map(self) -> dict[str, str]:
        return _pairs_to_dict(self.attributes, "TraversalRequestDTO.attributes")


@dataclass(frozen=True, slots=True)
class TraversalStepDTO:
    """One recorded step of a traversal - enough to replay the whole path."""

    tile_id: str
    rule_descriptor: TraversalRuleDescriptorDTO
    request_id: str
    eligible_children: Tuple[
        Tuple[str, str], ...
    ]  # (target_id, order_key) in declared order
    selected_child: Optional[str]
    tie_candidates: Tuple[str, ...] = field(default_factory=tuple)
    tie_resolution: str = ""
    failure_condition: Optional[str] = None
    spec_version: str = SPEC_VERSION


@dataclass(frozen=True, slots=True)
class TraversalFailureDTO:
    """A traversal failure represented as data, not an exception."""

    failure_code: str
    message: str
    at_tile_id: str
    spec_version: str = SPEC_VERSION


@dataclass(frozen=True, slots=True)
class TraversalResultDTO:
    """The outcome of one traversal: success or failure, with full trace."""

    success: bool
    final_leaf_id: Optional[str]
    final_artifact_kind: Optional[str]
    final_artifact_id: Optional[str]
    steps: Tuple[TraversalStepDTO, ...]
    failure: Optional[TraversalFailureDTO] = None
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        if self.success and self.failure is not None:
            raise VPMValidationError(
                "TraversalResultDTO cannot be successful and carry a failure"
            )
        if not self.success and self.failure is None:
            raise VPMValidationError(
                "TraversalResultDTO must carry a failure when not successful"
            )


@dataclass(frozen=True, slots=True)
class TraversalReceiptDTO:
    """An identified, replayable record of one traversal."""

    receipt_id: str
    hierarchy_id: str
    request: TraversalRequestDTO
    result: TraversalResultDTO
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.hierarchy_id,
            "TraversalReceiptDTO.hierarchy_id must be a sha256: digest",
        )
        expected_id = sha256_digest(
            canonical_json_bytes(receipt_identity_payload(self))
        )
        if self.receipt_id != expected_id:
            raise VPMValidationError(
                "TraversalReceiptDTO.receipt_id does not match its own canonical content"
            )


def _step_identity_payload(step: TraversalStepDTO) -> dict:
    return {
        "tile_id": step.tile_id,
        "rule_descriptor": {
            "rule_kind": step.rule_descriptor.rule_kind,
            "parameters": _pairs_to_dict(
                step.rule_descriptor.parameters, "TraversalRuleDescriptorDTO.parameters"
            ),
        },
        "request_id": step.request_id,
        "eligible_children": [list(pair) for pair in step.eligible_children],
        "selected_child": step.selected_child,
        "tie_candidates": list(step.tie_candidates),
        "tie_resolution": step.tie_resolution,
        "failure_condition": step.failure_condition,
    }


def _failure_identity_payload(failure: Optional[TraversalFailureDTO]) -> Optional[dict]:
    if failure is None:
        return None
    return {
        "failure_code": failure.failure_code,
        "message": failure.message,
        "at_tile_id": failure.at_tile_id,
    }


def _result_identity_payload(result: TraversalResultDTO) -> dict:
    return {
        "success": result.success,
        "final_leaf_id": result.final_leaf_id,
        "final_artifact_kind": result.final_artifact_kind,
        "final_artifact_id": result.final_artifact_id,
        "steps": [_step_identity_payload(step) for step in result.steps],
        "failure": _failure_identity_payload(result.failure),
    }


def _receipt_identity_payload_fields(
    *,
    hierarchy_id: str,
    request: TraversalRequestDTO,
    result: TraversalResultDTO,
    spec_version: str = SPEC_VERSION,
) -> dict:
    return {
        "spec_version": spec_version,
        "hierarchy_id": hierarchy_id,
        "request": {
            "request_id": request.request_id,
            "attributes": _pairs_to_dict(
                request.attributes, "TraversalRequestDTO.attributes"
            ),
        },
        "result": _result_identity_payload(result),
    }


def receipt_identity_payload(receipt: TraversalReceiptDTO) -> dict:
    """The exact canonical payload `receipt_id` covers.

    Includes every field that distinguishes one traversal outcome from
    another - not just the final leaf id, but the final artifact identity,
    every step's rule descriptor, eligible children, tie candidates, and
    the full failure detail. Two materially different traversals (e.g.
    same final leaf id but a different bound artifact, or a different
    failure reason) must never share a `receipt_id`.
    """
    return _receipt_identity_payload_fields(
        hierarchy_id=receipt.hierarchy_id,
        request=receipt.request,
        result=receipt.result,
        spec_version=receipt.spec_version,
    )


def build_traversal_receipt(
    *, hierarchy_id: str, request: TraversalRequestDTO, result: TraversalResultDTO
) -> TraversalReceiptDTO:
    payload = _receipt_identity_payload_fields(
        hierarchy_id=hierarchy_id, request=request, result=result
    )
    receipt_id = sha256_digest(canonical_json_bytes(payload))
    return TraversalReceiptDTO(
        receipt_id=receipt_id, hierarchy_id=hierarchy_id, request=request, result=result
    )
