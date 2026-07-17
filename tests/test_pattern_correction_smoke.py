from __future__ import annotations

from zeromodel import PatternAnalysisSpec, PatternDiscoveryArtifacts


def test_pattern_correction_public_surface() -> None:
    default = PatternAnalysisSpec()
    stricter = PatternAnalysisSpec(alpha=0.01)

    assert default.alpha == 0.05
    assert default.digest != stricter.digest
    assert PatternDiscoveryArtifacts.__name__ == "PatternDiscoveryArtifacts"
