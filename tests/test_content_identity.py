from __future__ import annotations

import numpy as np

from zeromodel.content_identity import (
    PROTOTYPE_UNIVERSE_IDENTITY_VERSION,
    UnresolvedArtifactIdentity,
    array_content_digest,
    prototype_universe_identity,
)
from zeromodel.visual_address import ImageObservation


def _prototypes() -> dict[str, tuple[str, str, str, ImageObservation]]:
    frame_a = np.zeros((2, 2), dtype=np.uint8)
    frame_b = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    obs_a = ImageObservation(frame_a, source_id="obs-a")
    obs_b = ImageObservation(frame_b, source_id="obs-b")
    return {
        "obs-b": ("row-b", "RIGHT", obs_b.raw_digest, obs_b),
        "obs-a": ("row-a", "LEFT", obs_a.raw_digest, obs_a),
    }


def test_prototype_identity_is_stable_across_object_reconstruction_and_order() -> None:
    original = _prototypes()
    reversed_order = dict(reversed(list(_prototypes().items())))
    identity_a = prototype_universe_identity(prototypes=original, policy_artifact_id="policy", source_scope="scope")
    identity_b = prototype_universe_identity(prototypes=reversed_order, policy_artifact_id="policy", source_scope="scope")
    assert identity_a.version == PROTOTYPE_UNIVERSE_IDENTITY_VERSION
    assert identity_a.digest == identity_b.digest
    assert identity_a.row_ids == ("row-a", "row-b")


def test_prototype_identity_changes_on_mutation_row_and_scope() -> None:
    prototypes = _prototypes()
    baseline = prototype_universe_identity(prototypes=prototypes, policy_artifact_id="policy", source_scope="scope")
    mutated_pixels = _prototypes()
    mutated_pixels["obs-a"][3].pixels.flags.writeable = True
    mutated_pixels["obs-a"][3].pixels[0, 0] = 1
    after_pixel = prototype_universe_identity(prototypes=mutated_pixels, policy_artifact_id="policy", source_scope="scope")
    changed_row = _prototypes()
    changed_row["obs-a"] = ("row-z", "LEFT", changed_row["obs-a"][2], changed_row["obs-a"][3])
    after_row = prototype_universe_identity(prototypes=changed_row, policy_artifact_id="policy", source_scope="scope")
    after_scope = prototype_universe_identity(prototypes=prototypes, policy_artifact_id="policy", source_scope="other")
    assert len({baseline.digest, after_pixel.digest, after_row.digest, after_scope.digest}) == 4


def test_array_content_digest_binds_dtype() -> None:
    as_u8 = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    as_i16 = as_u8.astype(np.int16)
    assert array_content_digest(as_u8) != array_content_digest(as_i16)


def test_unresolved_identity_cannot_use_sha256_prefix() -> None:
    try:
        UnresolvedArtifactIdentity("sha256:not-real", "bad")
    except Exception:
        pass
    else:
        raise AssertionError("expected unresolved identity validation failure")
