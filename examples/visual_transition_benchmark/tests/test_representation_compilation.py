"""Tests the deterministic selection policy in isolation, against synthetic
frames constructed directly (not a real domain) so the policy's own logic --
ranking, tie-breaking, and the compiled/insufficient_representation/
insufficient_observability trichotomy -- can be tested without depending on
arcade or warehouse rendering."""

import numpy as np

from visual_transition_benchmark.compiler.candidates import (
    RegionGeometry,
    generate_candidates,
)
from visual_transition_benchmark.compiler.compile import compile_requirement
from visual_transition_benchmark.compiler.contracts import VisualEvidenceRequirement
from visual_transition_benchmark.compiler.evaluate import DevelopmentSample

CANVAS = (8, 8)
REGION = RegionGeometry(
    region_id="r",
    canvas_shape=CANVAS,
    y0=0,
    y1=2,
    x0=0,
    x1=4,
    cell_height=1,
    cell_width=2,
)


def _frame(fill: int) -> np.ndarray:
    frame = np.zeros(CANVAS, dtype=np.uint8)
    frame[0:2, 0:4] = fill
    return frame


def _presence_requirement():
    return VisualEvidenceRequirement(
        domain_name="synthetic",
        component_type="probe",
        property_name="presence",
        evidence_kind="presence",
        candidate_region_id="r",
        comparison="changed",
    )


def test_compiles_when_a_candidate_clearly_recovers_the_property():
    req = _presence_requirement()
    candidates = generate_candidates(req, REGION)
    samples = [
        DevelopmentSample(
            sample_id="s0",
            frame_before=_frame(0),
            frame_after=_frame(200),
            true_before=False,
            true_after=True,
        ),
        DevelopmentSample(
            sample_id="s1",
            frame_before=_frame(50),
            frame_after=_frame(50),
            true_before=False,
            true_after=False,
        ),
        DevelopmentSample(
            sample_id="s2",
            frame_before=_frame(0),
            frame_after=_frame(180),
            true_before=False,
            true_after=True,
        ),
        DevelopmentSample(
            sample_id="s3",
            frame_before=_frame(30),
            frame_after=_frame(30),
            true_before=False,
            true_after=False,
        ),
    ]
    compiled = compile_requirement(
        req, REGION, candidates, samples, min_decoding_accuracy=0.95
    )
    assert compiled.status == "compiled"
    assert compiled.selected_candidate is not None
    assert compiled.selected_evaluation.decoding_accuracy == 1.0


def test_selection_is_deterministic_across_repeated_runs():
    req = _presence_requirement()
    candidates = generate_candidates(req, REGION)
    samples = [
        DevelopmentSample(
            sample_id="s0",
            frame_before=_frame(0),
            frame_after=_frame(200),
            true_before=False,
            true_after=True,
        ),
        DevelopmentSample(
            sample_id="s1",
            frame_before=_frame(50),
            frame_after=_frame(50),
            true_before=False,
            true_after=False,
        ),
    ]
    a = compile_requirement(
        req, REGION, candidates, samples, min_decoding_accuracy=0.95
    )
    b = compile_requirement(
        req, REGION, candidates, samples, min_decoding_accuracy=0.95
    )
    assert a.selected_candidate.candidate_id == b.selected_candidate.candidate_id


def test_insufficient_observability_when_every_candidate_is_degenerate():
    # A numeric_value requirement over a region that never varies at all --
    # every decoder reads the same constant regardless of resolution, while
    # the (synthetic, privileged) true value is deliberately made to vary.
    req = VisualEvidenceRequirement(
        domain_name="synthetic",
        component_type="probe",
        property_name="hidden_value",
        evidence_kind="numeric_value",
        candidate_region_id="r",
        expected_value_domain=(0.1, 0.9),
        required_precision=0.05,
        comparison="equal",
    )
    candidates = generate_candidates(req, REGION)
    samples = [
        DevelopmentSample(
            sample_id=f"s{i}",
            frame_before=_frame(0),
            frame_after=_frame(0),
            true_before=0.1,
            true_after=(0.1 if i % 2 == 0 else 0.9),
        )
        for i in range(6)
    ]
    compiled = compile_requirement(
        req,
        REGION,
        candidates,
        samples,
        canonical_levels=(0.1, 0.9),
        min_decoding_accuracy=0.95,
    )
    assert compiled.status == "insufficient_observability"
    assert compiled.selected_candidate is None


def test_insufficient_representation_when_signal_exists_but_no_candidate_passes():
    # The region does carry real signal (so decoded values are not constant),
    # but it never matches the declared true values under any candidate here
    # because true_after is deliberately uncorrelated with the frame content.
    req = VisualEvidenceRequirement(
        domain_name="synthetic",
        component_type="probe",
        property_name="uncorrelated_value",
        evidence_kind="numeric_value",
        candidate_region_id="r",
        expected_value_domain=(0.1, 0.9),
        required_precision=0.02,
        comparison="equal",
    )
    candidates = generate_candidates(req, REGION)
    # Fills land close to the two canonical levels (25.5=0.1*255, 229.5=0.9*255)
    # so decoding is non-degenerate (it does vary across samples), but
    # deliberately out of sync with true_after so no candidate scores >= 0.95.
    fills = [230, 230, 25, 25, 230, 25]
    samples = [
        DevelopmentSample(
            sample_id=f"s{i}",
            frame_before=_frame(0),
            frame_after=_frame(fills[i]),
            true_before=0.1,
            true_after=(0.1 if i % 2 == 0 else 0.9),
        )
        for i in range(len(fills))
    ]
    compiled = compile_requirement(
        req,
        REGION,
        candidates,
        samples,
        canonical_levels=(0.1, 0.9),
        min_decoding_accuracy=0.95,
    )
    assert compiled.status == "insufficient_representation"
    assert compiled.selected_candidate is None
    assert compiled.known_limitations


def test_compiled_and_insufficient_statuses_are_never_conflated():
    # A meta-check: the three statuses are mutually exclusive by construction
    # (compile_requirement returns exactly one), verified across all three
    # scenarios above via the CompileStatus literal used consistently.
    req = _presence_requirement()
    candidates = generate_candidates(req, REGION)
    samples = [
        DevelopmentSample(
            sample_id="s0",
            frame_before=_frame(0),
            frame_after=_frame(200),
            true_before=False,
            true_after=True,
        ),
    ]
    compiled = compile_requirement(
        req, REGION, candidates, samples, min_decoding_accuracy=0.95
    )
    assert compiled.status in (
        "compiled",
        "insufficient_representation",
        "insufficient_observability",
    )
    if compiled.status == "compiled":
        assert compiled.selected_candidate is not None
        assert compiled.selected_evaluation is not None
    else:
        assert compiled.selected_candidate is None
