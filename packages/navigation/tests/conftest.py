from __future__ import annotations

import pytest

from zeromodel.artifacts import ArtifactRef, InMemoryArtifactStore, sha256_digest
from zeromodel.navigation.dto import HierarchyCompilerSpecDTO

CORPUS_ARTIFACT_KIND = "policy-snapshot"


@pytest.fixture
def artifact_store() -> InMemoryArtifactStore:
    return InMemoryArtifactStore()


@pytest.fixture
def compiler_spec() -> HierarchyCompilerSpecDTO:
    return HierarchyCompilerSpecDTO(
        compiler_id="zeromodel-navigation-reference-compiler",
        compiler_version="1.0.0",
        corpus_id="corpus-fleet-policies",
        corpus_artifact_kind=CORPUS_ARTIFACT_KIND,
        leaf_semantics="terminal-policy-artifact",
        max_children_per_tile=3,
        max_depth=8,
        tie_rule="lowest_order_key",
        failure_rule="fail-closed-no-match",
        navigation_rule_contract="fixed-key-or-declared-priority/v1",
    )


def make_source_artifacts(count: int, *, kind: str = CORPUS_ARTIFACT_KIND) -> tuple:
    return tuple(
        ArtifactRef(
            artifact_kind=kind,
            artifact_id=sha256_digest(f"artifact-payload-{i}".encode()),
        )
        for i in range(count)
    )
