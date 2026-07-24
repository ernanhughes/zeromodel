"""Tests for `examples/arcade_png_interventions.py`: recipe identity,
determinism, and pixel-level visual distinction between representation
variants. No network, no Ollama - pure PIL pixel operations only.
"""

from __future__ import annotations

import numpy as np
import pytest

import examples.local_model_zero_arcade_test as arcade
from examples.arcade_png_interventions import (
    ALL_VARIANTS,
    COMBINED_VARIANT,
    COOLDOWN_DUAL_VARIANT,
    COOLDOWN_REDUNDANT_VARIANT,
    COOLDOWN_SHAPE_VARIANT,
    COOLDOWN_TEXT_VARIANT,
    FOOTER_ONLY_VARIANT,
    FOOTER_TOP,
    IMG_HEIGHT,
    IMG_WIDTH,
    LABELLED_VARIANT,
    LANE_ENHANCED_VARIANT,
    LANE_NUMERALS_VARIANT,
    LANES,
    LEFT_MARGIN,
    PRIMARY_COOLDOWN_BOX,
    RIGHT_MARGIN,
    SECONDARY_COOLDOWN_BOX,
    SEMANTIC_LABELLED_VARIANT,
    SEMANTIC_VARIANTS,
    UNLABELLED_VARIANT,
    ArcadePngInterventionRecipe,
    ArcadePngOperationSpec,
    apply_recipe,
    build_recipe,
)

READY_STATE = arcade.ArcadeState(3, 3, 0)
BLOCKED_STATE = arcade.ArcadeState(3, 3, 1)


def _final_image(recipe: ArcadePngInterventionRecipe, state: arcade.ArcadeState):
    base_image = arcade.render(state, recipe.base_render_mode)
    steps = apply_recipe(recipe, base_image)
    return steps[-1][0], steps


class TestRecipeIdentity:
    def test_every_variant_has_a_declared_recipe(self) -> None:
        for variant_id in ALL_VARIANTS:
            if variant_id == COMBINED_VARIANT:
                continue
            recipe = build_recipe(variant_id)
            assert recipe.variant_id == variant_id
            assert recipe.recipe_id.startswith("sha256:")

    def test_unknown_variant_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown representation variant"):
            build_recipe("does-not-exist-v1")

    def test_combined_requires_component_args(self) -> None:
        with pytest.raises(ValueError, match="combined-cooldown"):
            build_recipe(COMBINED_VARIANT)
        with pytest.raises(ValueError, match="combined-lane"):
            build_recipe(COMBINED_VARIANT, combined_cooldown=COOLDOWN_DUAL_VARIANT)
        with pytest.raises(ValueError):
            build_recipe(COMBINED_VARIANT, combined_cooldown="not-a-cooldown-variant")

    def test_combined_recipe_builds_with_valid_component_args(self) -> None:
        recipe = build_recipe(
            COMBINED_VARIANT,
            combined_cooldown=COOLDOWN_DUAL_VARIANT,
            combined_lane=LANE_ENHANCED_VARIANT,
        )
        assert recipe.variant_id == COMBINED_VARIANT
        operation_names = [op.operation for op in recipe.operations]
        assert "cooldown_dual_overlay" in operation_names
        assert "lane_separator_enhance" in operation_names

    def test_operation_order_affects_identity(self) -> None:
        op_a = ArcadePngOperationSpec("op_a", "v1", {"x": 1})
        op_b = ArcadePngOperationSpec("op_b", "v1", {"y": 2})
        forward = ArcadePngInterventionRecipe.build(
            variant_id="order-test",
            base_render_mode="unlabelled",
            operations=(op_a, op_b),
        )
        backward = ArcadePngInterventionRecipe.build(
            variant_id="order-test",
            base_render_mode="unlabelled",
            operations=(op_b, op_a),
        )
        assert forward.recipe_id != backward.recipe_id

    def test_parameter_changes_affect_identity(self) -> None:
        op1 = ArcadePngOperationSpec("op", "v1", {"x": 1})
        op2 = ArcadePngOperationSpec("op", "v1", {"x": 2})
        recipe1 = ArcadePngInterventionRecipe.build(
            variant_id="param-test", base_render_mode="unlabelled", operations=(op1,)
        )
        recipe2 = ArcadePngInterventionRecipe.build(
            variant_id="param-test", base_render_mode="unlabelled", operations=(op2,)
        )
        assert recipe1.recipe_id != recipe2.recipe_id

    def test_identical_recipe_produces_identical_png_bytes(self) -> None:
        recipe = build_recipe(COOLDOWN_DUAL_VARIANT)
        base_image = arcade.render(BLOCKED_STATE, recipe.base_render_mode)
        steps_a = apply_recipe(recipe, base_image)
        steps_b = apply_recipe(recipe, base_image)
        assert steps_a[-1][1] == steps_b[-1][1]

    def test_recipe_id_round_trips_through_to_dict(self) -> None:
        recipe = build_recipe(LANE_ENHANCED_VARIANT)
        payload = recipe.to_dict()
        assert payload["recipe_id"] == recipe.recipe_id
        assert payload["variant_id"] == LANE_ENHANCED_VARIANT

    def test_tampered_recipe_id_rejected(self) -> None:
        recipe = build_recipe(COOLDOWN_SHAPE_VARIANT)
        with pytest.raises(ValueError, match="recipe id mismatch"):
            ArcadePngInterventionRecipe(
                version=recipe.version,
                variant_id=recipe.variant_id,
                base_render_mode=recipe.base_render_mode,
                operations=recipe.operations,
                metadata=recipe.metadata,
                recipe_id="sha256:" + "0" * 64,
            )


class TestNoOpRejection:
    def test_no_op_transform_is_rejected(self) -> None:
        """A duplicate operation whose source and target positions are the
        same box copies a region onto itself - a genuine byte-identical
        no-op - and must be rejected."""
        noop = ArcadePngOperationSpec(
            operation="cooldown_marker_duplicate",
            operation_version="v1",
            parameters={
                "source_position": list(PRIMARY_COOLDOWN_BOX),
                "target_position": list(PRIMARY_COOLDOWN_BOX),
            },
        )
        recipe = ArcadePngInterventionRecipe.build(
            variant_id="self-duplicate",
            base_render_mode="unlabelled",
            operations=(noop,),
        )
        base_image = arcade.render(READY_STATE, recipe.base_render_mode)
        with pytest.raises(ValueError, match="produced no pixel change"):
            apply_recipe(recipe, base_image)

    def test_unknown_operation_rejected(self) -> None:
        bad_op = ArcadePngOperationSpec("does_not_exist", "v1", {})
        recipe = ArcadePngInterventionRecipe.build(
            variant_id="unknown-op", base_render_mode="unlabelled", operations=(bad_op,)
        )
        base_image = arcade.render(READY_STATE, recipe.base_render_mode)
        with pytest.raises(ValueError, match="unknown intervention operation"):
            apply_recipe(recipe, base_image)


class TestPixelDistinction:
    def test_labelled_and_unlabelled_differ(self) -> None:
        labelled_final, _ = _final_image(build_recipe(LABELLED_VARIANT), READY_STATE)
        unlabelled_final, _ = _final_image(
            build_recipe(UNLABELLED_VARIANT), READY_STATE
        )
        assert labelled_final.tobytes() != unlabelled_final.tobytes()

    def test_cooldown_shape_differs_between_ready_and_blocked(self) -> None:
        recipe = build_recipe(COOLDOWN_SHAPE_VARIANT)
        ready_final, _ = _final_image(recipe, READY_STATE)
        blocked_final, _ = _final_image(recipe, BLOCKED_STATE)
        ready_crop = ready_final.crop(PRIMARY_COOLDOWN_BOX).tobytes()
        blocked_crop = blocked_final.crop(PRIMARY_COOLDOWN_BOX).tobytes()
        assert ready_crop != blocked_crop

    def test_cooldown_dual_changes_both_shape_and_colour_channel(self) -> None:
        shape_recipe = build_recipe(COOLDOWN_SHAPE_VARIANT)
        dual_recipe = build_recipe(COOLDOWN_DUAL_VARIANT)
        shape_final, _ = _final_image(shape_recipe, BLOCKED_STATE)
        dual_final, _ = _final_image(dual_recipe, BLOCKED_STATE)
        shape_crop = shape_final.crop(PRIMARY_COOLDOWN_BOX)
        dual_crop = dual_final.crop(PRIMARY_COOLDOWN_BOX)
        # Shape-only has no fill (background only); dual fills with the red
        # blocked colour - the dual crop must contain red pixels the
        # shape-only crop does not.
        assert dual_crop.tobytes() != shape_crop.tobytes()
        dual_colors = {
            rgb
            for _count, rgb in dual_crop.convert("RGB").getcolors(maxcolors=1_000_000)
        }
        assert (242, 74, 74) in dual_colors

    def test_redundant_has_two_identical_markers(self) -> None:
        recipe = build_recipe(COOLDOWN_REDUNDANT_VARIANT)
        final_image, _ = _final_image(recipe, BLOCKED_STATE)
        primary = final_image.crop(PRIMARY_COOLDOWN_BOX).tobytes()
        secondary = final_image.crop(SECONDARY_COOLDOWN_BOX).tobytes()
        assert primary == secondary
        base_unlabelled, _ = _final_image(
            build_recipe(UNLABELLED_VARIANT), BLOCKED_STATE
        )
        base_secondary = base_unlabelled.crop(SECONDARY_COOLDOWN_BOX).tobytes()
        assert secondary != base_secondary

    def test_lane_enhanced_changes_only_declared_regions(self) -> None:
        baseline_final, _ = _final_image(
            build_recipe(UNLABELLED_VARIANT), BLOCKED_STATE
        )
        enhanced_final, _ = _final_image(
            build_recipe(LANE_ENHANCED_VARIANT), BLOCKED_STATE
        )
        assert baseline_final.tobytes() != enhanced_final.tobytes()
        baseline_cooldown = baseline_final.crop(PRIMARY_COOLDOWN_BOX).tobytes()
        enhanced_cooldown = enhanced_final.crop(PRIMARY_COOLDOWN_BOX).tobytes()
        assert baseline_cooldown == enhanced_cooldown

    def test_same_state_and_recipe_produces_identical_bytes(self) -> None:
        recipe = build_recipe(LANE_ENHANCED_VARIANT)
        first, _ = _final_image(recipe, READY_STATE)
        second, _ = _final_image(recipe, READY_STATE)
        assert first.tobytes() == second.tobytes()

    def test_source_and_result_digests_differ_when_pixels_change(self) -> None:
        recipe = build_recipe(COOLDOWN_DUAL_VARIANT)
        base_image = arcade.render(BLOCKED_STATE, recipe.base_render_mode)
        steps = apply_recipe(recipe, base_image)
        assert len(steps) == 2
        assert steps[0][1] != steps[1][1]

    def test_reference_variants_produce_a_single_step(self) -> None:
        base_image = arcade.render(READY_STATE, "labelled")
        steps = apply_recipe(build_recipe(LABELLED_VARIANT), base_image)
        assert len(steps) == 1


def _lane_center_x_positions() -> list[float]:
    lane_w = (IMG_WIDTH - LEFT_MARGIN - RIGHT_MARGIN) / LANES
    return [LEFT_MARGIN + (index + 0.5) * lane_w for index in range(LANES)]


def _lane_boundary_x_positions() -> list[float]:
    lane_w = (IMG_WIDTH - LEFT_MARGIN - RIGHT_MARGIN) / LANES
    return [LEFT_MARGIN + index * lane_w for index in range(LANES + 1)]


def _footer_band_diff_mask(base, candidate) -> np.ndarray:
    base_arr = np.asarray(base.convert("L"))[FOOTER_TOP:IMG_HEIGHT, :]
    candidate_arr = np.asarray(candidate.convert("L"))[FOOTER_TOP:IMG_HEIGHT, :]
    return np.any(base_arr != candidate_arr, axis=0)


def _count_column_clusters(diff_mask: np.ndarray) -> int:
    clusters = 0
    previous = False
    for value in diff_mask:
        if value and not previous:
            clusters += 1
        previous = bool(value)
    return clusters


COOLDOWN_TEXT_REGION = (60, 20, 300, 40)


class TestSemanticAnnotationVariants:
    """Stage 2F: footer geometry / lane numerals / cooldown text isolation.

    See `docs/reviews/stage-2f-semantic-annotation-ablation.md` for why these
    three factors were isolated from the previously bundled `labelled-v1`
    representation.
    """

    def test_all_semantic_variants_are_declared(self) -> None:
        for variant_id in SEMANTIC_VARIANTS:
            assert variant_id in ALL_VARIANTS
            recipe = build_recipe(variant_id)
            assert recipe.variant_id == variant_id
            assert recipe.recipe_id.startswith("sha256:")

    def test_each_semantic_variant_recipe_is_deterministic(self) -> None:
        for variant_id in SEMANTIC_VARIANTS:
            assert (
                build_recipe(variant_id).recipe_id == build_recipe(variant_id).recipe_id
            )

    def test_footer_only_declares_only_the_footer_operation(self) -> None:
        recipe = build_recipe(FOOTER_ONLY_VARIANT)
        assert {op.operation for op in recipe.operations} == {"footer_reserved_area"}

    def test_lane_numerals_declares_footer_and_numeral_operations_only(self) -> None:
        recipe = build_recipe(LANE_NUMERALS_VARIANT)
        assert {op.operation for op in recipe.operations} == {
            "footer_reserved_area",
            "lane_numerals_overlay",
        }

    def test_cooldown_text_declares_only_the_cooldown_text_operation(self) -> None:
        recipe = build_recipe(COOLDOWN_TEXT_VARIANT)
        assert {op.operation for op in recipe.operations} == {"cooldown_text_overlay"}

    def test_semantic_labelled_declares_all_three_operations(self) -> None:
        recipe = build_recipe(SEMANTIC_LABELLED_VARIANT)
        assert {op.operation for op in recipe.operations} == {
            "footer_reserved_area",
            "lane_numerals_overlay",
            "cooldown_text_overlay",
        }

    def test_footer_only_touches_nothing_above_the_footer_band(self) -> None:
        baseline, _ = _final_image(build_recipe(UNLABELLED_VARIANT), READY_STATE)
        footer_only, _ = _final_image(build_recipe(FOOTER_ONLY_VARIANT), READY_STATE)
        assert baseline.tobytes() != footer_only.tobytes()
        above_baseline = baseline.crop((0, 0, IMG_WIDTH, FOOTER_TOP)).tobytes()
        above_footer = footer_only.crop((0, 0, IMG_WIDTH, FOOTER_TOP)).tobytes()
        assert above_baseline == above_footer
        cooldown_baseline = baseline.crop(COOLDOWN_TEXT_REGION).tobytes()
        cooldown_footer = footer_only.crop(COOLDOWN_TEXT_REGION).tobytes()
        assert cooldown_baseline == cooldown_footer

    def test_footer_does_not_occlude_the_tank_sprite(self) -> None:
        """The tank sprite's lowest point (`bottom - 45 + 33` = 476 for every
        lane and state) must stay clear of the footer band - a regression
        guard for `FOOTER_TOP`'s geometry."""
        baseline, _ = _final_image(build_recipe(UNLABELLED_VARIANT), READY_STATE)
        footer_only, _ = _final_image(build_recipe(FOOTER_ONLY_VARIANT), READY_STATE)
        tank_region = (0, 400, IMG_WIDTH, 478)
        assert (
            baseline.crop(tank_region).tobytes()
            == footer_only.crop(tank_region).tobytes()
        )

    def test_lane_numerals_draws_exactly_seven_glyph_clusters_not_eight(self) -> None:
        """Guards the Stage 2E lesson directly: the real-provider
        `lane-enhanced-v1` result emphasized the eight boundary lines between
        lanes and regressed (one rejected `TANK_COLUMN: 7` response, more
        action-changing errors - see
        `docs/results/controlled-png-representation-v1/`). Lane semantics
        here must be drawn at the seven lane *centres* only."""
        centres = _lane_center_x_positions()
        boundaries = _lane_boundary_x_positions()
        assert len(centres) == LANES == 7
        assert len(boundaries) == LANES + 1 == 8

        footer_only, _ = _final_image(build_recipe(FOOTER_ONLY_VARIANT), READY_STATE)
        lane_numerals, _ = _final_image(
            build_recipe(LANE_NUMERALS_VARIANT), READY_STATE
        )
        diff_mask = _footer_band_diff_mask(footer_only, lane_numerals)

        assert _count_column_clusters(diff_mask) == 7

        for x in centres:
            lo, hi = max(0, round(x) - 6), min(diff_mask.shape[0], round(x) + 6)
            assert np.any(diff_mask[lo:hi]), (
                f"expected numeral content near centre x={x}"
            )

    def test_cooldown_text_has_no_lane_numerals(self) -> None:
        baseline, _ = _final_image(build_recipe(UNLABELLED_VARIANT), BLOCKED_STATE)
        cooldown_text, _ = _final_image(
            build_recipe(COOLDOWN_TEXT_VARIANT), BLOCKED_STATE
        )
        assert not np.any(_footer_band_diff_mask(baseline, cooldown_text))
        cooldown_baseline = baseline.crop(COOLDOWN_TEXT_REGION).tobytes()
        cooldown_text_crop = cooldown_text.crop(COOLDOWN_TEXT_REGION).tobytes()
        assert cooldown_baseline != cooldown_text_crop

    def test_cooldown_text_reflects_ready_and_blocked_states(self) -> None:
        recipe = build_recipe(COOLDOWN_TEXT_VARIANT)
        ready_final, _ = _final_image(recipe, READY_STATE)
        blocked_final, _ = _final_image(recipe, BLOCKED_STATE)
        ready_crop = ready_final.crop(COOLDOWN_TEXT_REGION).tobytes()
        blocked_crop = blocked_final.crop(COOLDOWN_TEXT_REGION).tobytes()
        assert ready_crop != blocked_crop

    def test_semantic_labelled_contains_all_three_components(self) -> None:
        baseline, _ = _final_image(build_recipe(UNLABELLED_VARIANT), BLOCKED_STATE)
        semantic, _ = _final_image(
            build_recipe(SEMANTIC_LABELLED_VARIANT), BLOCKED_STATE
        )
        diff_mask = _footer_band_diff_mask(baseline, semantic)
        assert np.any(diff_mask)
        for x in _lane_center_x_positions():
            lo, hi = max(0, round(x) - 6), min(diff_mask.shape[0], round(x) + 6)
            assert np.any(diff_mask[lo:hi])
        cooldown_baseline = baseline.crop(COOLDOWN_TEXT_REGION).tobytes()
        cooldown_semantic = semantic.crop(COOLDOWN_TEXT_REGION).tobytes()
        assert cooldown_baseline != cooldown_semantic

    def test_semantic_variants_never_mutate_the_input_image(self) -> None:
        for variant_id in SEMANTIC_VARIANTS:
            recipe = build_recipe(variant_id)
            base_image = arcade.render(READY_STATE, recipe.base_render_mode)
            original_bytes = base_image.tobytes()
            apply_recipe(recipe, base_image)
            assert base_image.tobytes() == original_bytes
