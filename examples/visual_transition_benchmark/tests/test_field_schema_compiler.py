"""Tests the compiler in isolation, and the explicit "representation
compilation" claim from the cross-domain experiment: does the same compiler,
given declared evidence requirements matching what stages 1 and 2 hand-built,
reproduce their exact tile sizes?
"""

from visual_transition_benchmark import zeromodel_adapter as zm
from visual_transition_benchmark import value_contracts as vc
from visual_transition_benchmark.compilation.evidence_requirements import (
    VisualEvidenceRequirement,
)
from visual_transition_benchmark.compilation.field_schema_compiler import (
    FieldSchemaCompilationError,
    compile_field_schema,
)
from visual_transition_benchmark.domains.warehouse import contracts as wc


def test_compiler_reproduces_stage1_coarse_tile_size_from_presence_requirements():
    # Stage 1 hand-built a 4x1px schema because presence detection only needs
    # to know "did this named band change", at 1 field per game column.
    requirements = (
        VisualEvidenceRequirement(
            component="tank",
            property_name="presence",
            evidence_kind="presence",
            region=(11, 14, 0, zm.WIDTH_PX),
            required_resolution=(1, 4),
            aggregation="mean",
        ),
    )
    compiled = compile_field_schema((zm.FRAME_HEIGHT, zm.WIDTH_PX), requirements)
    assert (compiled.tile_height, compiled.tile_width) == (1, 4)
    assert len(compiled.field_schema.fields) == len(zm.FIELD_SCHEMA.fields)


def test_compiler_reproduces_stage2_fine_tile_size_from_value_requirements():
    # Stage 2 needed 1x1px tiles because the cooldown indicator is only 2px
    # wide inside a 4px tile (the dilution bug). Declaring that requirement
    # should force the compiler to the same fine resolution stage 2 hand-fixed to.
    requirements = (
        VisualEvidenceRequirement(
            component="cooldown",
            property_name="level",
            evidence_kind="numeric_intensity",
            region=(7, 9, zm.WIDTH_PX - 3, zm.WIDTH_PX - 1),
            required_resolution=(1, 1),
            aggregation="mean",
        ),
    )
    compiled = compile_field_schema((zm.FRAME_HEIGHT, zm.WIDTH_PX), requirements)
    assert (compiled.tile_height, compiled.tile_width) == (1, 1)
    assert len(compiled.field_schema.fields) == len(vc.VALUE_FIELD_SCHEMA.fields)


def test_compiler_picks_the_finest_resolution_across_mixed_requirements():
    coarse = VisualEvidenceRequirement(
        component="a",
        property_name="presence",
        evidence_kind="presence",
        region=(0, 8, 0, 8),
        required_resolution=(4, 4),
        aggregation="mean",
    )
    fine = VisualEvidenceRequirement(
        component="b",
        property_name="value",
        evidence_kind="numeric_intensity",
        region=(0, 8, 0, 8),
        required_resolution=(1, 2),
        aggregation="mean",
    )
    compiled = compile_field_schema((8, 8), (coarse, fine))
    assert (compiled.tile_height, compiled.tile_width) == (1, 2)


def test_compiler_rejects_a_requirement_that_resolves_to_zero_fields():
    bad = VisualEvidenceRequirement(
        component="x",
        property_name="y",
        evidence_kind="presence",
        region=(0, 3, 0, 3),
        required_resolution=(4, 4),
        aggregation="mean",
    )
    try:
        compile_field_schema((8, 8), (bad,))
        assert False, "expected FieldSchemaCompilationError"
    except FieldSchemaCompilationError:
        pass


def test_warehouse_domain_actually_uses_the_compiled_schema():
    # The warehouse domain's own two requirements should compile to the
    # per-pixel resolution it was built and validated against.
    assert (wc.COMPILED.tile_height, wc.COMPILED.tile_width) == (1, 1)
    assert wc.COMPILED.canvas_shape == wc.CANVAS_SHAPE
