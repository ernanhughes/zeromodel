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


@pytest.fixture
def make_source_artifacts():
    """Factory fixture (not a plain module import - a bare `conftest`
    module name collides across package test directories once more than
    one package's tests are collected in the same pytest session, e.g. by
    `scripts/run_fast_tests.py`) building `count` distinct source
    ArtifactRefs.

    When `store` is given, the corresponding payload is actually stored
    under each ref - hierarchy closure now requires every source artifact
    a leaf binding points at to genuinely resolve through the Artifacts
    store, not merely have a well-formed digest.
    """

    def _make_source_artifacts(
        count: int,
        *,
        store: InMemoryArtifactStore | None = None,
        kind: str = CORPUS_ARTIFACT_KIND,
    ) -> tuple:
        refs = []
        for i in range(count):
            payload = f"artifact-payload-{i}".encode()
            if store is not None:
                ref = store.put(kind, payload)
            else:
                ref = ArtifactRef(
                    artifact_kind=kind, artifact_id=sha256_digest(payload)
                )
            refs.append(ref)
        return tuple(refs)

    return _make_source_artifacts
