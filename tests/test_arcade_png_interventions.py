"""Recipe identity, determinism, and pixel-isolation tests for arcade PNG variants."""

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
    IMG_HEIGHT,
    IMG_WIDTH,
    LABELLED_BOTTOM,
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
    png_bytes,
)

READY_STATE = arcade.ArcadeState(3, 3, 0)
BLOCKED_STATE = arcade.ArcadeState(3, 3, 1)
COOLDOWN_TEXT_REGION = (60, 20, 300, 42)
LANE_TEXT_REGION = (0, IMG_HEIGHT - 36, IMG_WIDTH, IMG_HEIGHT)


def _final_image(recipe: ArcadePngInterventionRecipe, state: arcade.ArcadeState):
    base = arcade.render(state, recipe.base_render_mode)
    steps = apply_recipe(recipe, base)
    return steps[-1][0], steps


def _lane_centres() -> list[float]:
    lane_width = (IMG_WIDTH - LEFT_MARGIN - RIGHT_MARGIN) / LANES
    return [LEFT_MARGIN + (index + 0.5) * lane_width for index in range(LANES)]


class TestRecipeIdentity:
    def test_every_declared_variant_builds(self) -> None:
        for variant_id in ALL_VARIANTS:
            if variant_id == COMBINED_VARIANT:
                continue
            recipe = build_recipe(variant_id)
            assert recipe.variant_id == variant_id
            assert recipe.recipe_id.startswith("sha256:")

    def test_unknown_variant_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown representation variant"):
            build_recipe("does-not-exist-v1")

    def test_combined_requires_named_components(self) -> None:
        with pytest.raises(ValueError, match="combined-cooldown"):
            build_recipe(COMBINED_VARIANT)
        with pytest.raises(ValueError, match="combined-lane"):
            build_recipe(COMBINED_VARIANT, combined_cooldown=COOLDOWN_DUAL_VARIANT)

    def test_combined_contains_both_component_families(self) -> None:
        recipe = build_recipe(
            COMBINED_VARIANT,
            combined_cooldown=COOLDOWN_DUAL_VARIANT,
            combined_lane=LANE_ENHANCED_VARIANT,
        )
        names = [operation.operation for operation in recipe.operations]
        assert "cooldown_dual_overlay" in names
        assert "lane_separator_enhance" in names

    def test_operation_order_changes_recipe_identity(self) -> None:
        first = ArcadePngOperationSpec("first", "v1", {"x": 1})
        second = ArcadePngOperationSpec("second", "v1", {"x": 2})
        forward = ArcadePngInterventionRecipe.build(
            variant_id="order",
            base_render_mode="unlabelled",
            operations=(first, second),
        )
        reverse = ArcadePngInterventionRecipe.build(
            variant_id="order",
            base_render_mode="unlabelled",
            operations=(second, first),
        )
        assert forward.recipe_id != reverse.recipe_id

    def test_parameter_change_changes_recipe_identity(self) -> None:
        one = ArcadePngInterventionRecipe.build(
            variant_id="parameters",
            base_render_mode="unlabelled",
            operations=(ArcadePngOperationSpec("op", "v1", {"x": 1}),),
        )
        two = ArcadePngInterventionRecipe.build(
            variant_id="parameters",
            base_render_mode="unlabelled",
            operations=(ArcadePngOperationSpec("op", "v1", {"x": 2}),),
        )
        assert one.recipe_id != two.recipe_id

    def test_recipe_id_tampering_is_rejected(self) -> None:
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


class TestStage2EOperations:
    def test_shape_encoding_differs_between_ready_and_blocked(self) -> None:
        recipe = build_recipe(COOLDOWN_SHAPE_VARIANT)
        ready, _ = _final_image(recipe, READY_STATE)
        blocked, _ = _final_image(recipe, BLOCKED_STATE)
        assert (
            ready.crop(PRIMARY_COOLDOWN_BOX).tobytes()
            != blocked.crop(PRIMARY_COOLDOWN_BOX).tobytes()
        )

    def test_dual_encoding_contains_blocked_red(self) -> None:
        final, _ = _final_image(build_recipe(COOLDOWN_DUAL_VARIANT), BLOCKED_STATE)
        colors = {
            color
            for _count, color in final.crop(PRIMARY_COOLDOWN_BOX)
            .convert("RGB")
            .getcolors(maxcolors=1_000_000)
        }
        assert (242, 74, 74) in colors

    def test_redundant_encoding_duplicates_the_marker(self) -> None:
        final, _ = _final_image(build_recipe(COOLDOWN_REDUNDANT_VARIANT), BLOCKED_STATE)
        assert (
            final.crop(PRIMARY_COOLDOWN_BOX).tobytes()
            == final.crop(SECONDARY_COOLDOWN_BOX).tobytes()
        )

    def test_lane_enhancement_does_not_touch_cooldown(self) -> None:
        baseline, _ = _final_image(build_recipe(UNLABELLED_VARIANT), BLOCKED_STATE)
        enhanced, _ = _final_image(build_recipe(LANE_ENHANCED_VARIANT), BLOCKED_STATE)
        assert baseline.tobytes() != enhanced.tobytes()
        assert (
            baseline.crop(PRIMARY_COOLDOWN_BOX).tobytes()
            == enhanced.crop(PRIMARY_COOLDOWN_BOX).tobytes()
        )

    def test_operations_are_deterministic(self) -> None:
        recipe = build_recipe(COOLDOWN_DUAL_VARIANT)
        base = arcade.render(BLOCKED_STATE, recipe.base_render_mode)
        assert apply_recipe(recipe, base)[-1][1] == apply_recipe(recipe, base)[-1][1]

    def test_no_op_operation_is_rejected(self) -> None:
        operation = ArcadePngOperationSpec(
            "cooldown_marker_duplicate",
            "v1",
            {
                "source_position": list(PRIMARY_COOLDOWN_BOX),
                "target_position": list(PRIMARY_COOLDOWN_BOX),
            },
        )
        recipe = ArcadePngInterventionRecipe.build(
            variant_id="no-op", base_render_mode="unlabelled", operations=(operation,)
        )
        with pytest.raises(ValueError, match="produced no pixel change"):
            apply_recipe(recipe, arcade.render(READY_STATE, "unlabelled"))

    def test_unknown_operation_is_rejected(self) -> None:
        recipe = ArcadePngInterventionRecipe.build(
            variant_id="unknown",
            base_render_mode="unlabelled",
            operations=(ArcadePngOperationSpec("missing", "v1", {}),),
        )
        with pytest.raises(ValueError, match="unknown intervention operation"):
            apply_recipe(recipe, arcade.render(READY_STATE, "unlabelled"))


class TestHierarchicalSemanticAblation:
    def test_semantic_variants_are_content_addressed_and_distinct(self) -> None:
        recipes = [build_recipe(variant_id) for variant_id in SEMANTIC_VARIANTS]
        assert len({recipe.recipe_id for recipe in recipes}) == len(recipes)

    def test_footer_only_uses_historical_labelled_geometry(self) -> None:
        recipe = build_recipe(FOOTER_ONLY_VARIANT)
        assert recipe.base_render_mode == "labelled"
        assert [operation.operation for operation in recipe.operations] == [
            "cooldown_text_erase",
            "lane_numerals_erase",
        ]
        final, _ = _final_image(recipe, READY_STATE)
        # The historical labelled grid terminates at y=464 rather than the
        # unlabelled y=488. Its horizontal grid line must remain present.
        assert np.any(
            np.asarray(final)[LABELLED_BOTTOM, :, :] != np.asarray(final)[0, 0, :]
        )

    def test_lane_numerals_keeps_labelled_geometry_and_removes_only_cooldown_text(
        self,
    ) -> None:
        recipe = build_recipe(LANE_NUMERALS_VARIANT)
        assert recipe.base_render_mode == "labelled"
        assert [operation.operation for operation in recipe.operations] == [
            "cooldown_text_erase"
        ]
        final, _ = _final_image(recipe, BLOCKED_STATE)
        labelled = arcade.render(BLOCKED_STATE, "labelled")
        assert (
            final.crop(LANE_TEXT_REGION).tobytes()
            == labelled.crop(LANE_TEXT_REGION).tobytes()
        )
        assert (
            final.crop(COOLDOWN_TEXT_REGION).tobytes()
            != labelled.crop(COOLDOWN_TEXT_REGION).tobytes()
        )

    def test_lane_numerals_encode_exactly_seven_region_centres(self) -> None:
        footer_only, _ = _final_image(build_recipe(FOOTER_ONLY_VARIANT), READY_STATE)
        numerals, _ = _final_image(build_recipe(LANE_NUMERALS_VARIANT), READY_STATE)
        base = np.asarray(footer_only.convert("L"))[IMG_HEIGHT - 36 :, :]
        candidate = np.asarray(numerals.convert("L"))[IMG_HEIGHT - 36 :, :]
        changed_columns = np.any(base != candidate, axis=0)
        clusters = 0
        active = False
        for changed in changed_columns:
            if changed and not active:
                clusters += 1
            active = bool(changed)
        assert clusters == 7
        for centre in _lane_centres():
            lo, hi = round(centre) - 6, round(centre) + 6
            assert np.any(changed_columns[lo:hi])

    def test_cooldown_text_uses_exact_historical_wording(self) -> None:
        recipe = build_recipe(COOLDOWN_TEXT_VARIANT)
        assert recipe.base_render_mode == "unlabelled"
        for state in (READY_STATE, BLOCKED_STATE):
            final, _ = _final_image(recipe, state)
            expected = arcade.render(state, "unlabelled")
            # Reproduce only the historical text on the unlabelled geometry.
            from PIL import ImageDraw

            draw = ImageDraw.Draw(expected)
            draw.text(
                (68, 27),
                "READY (cooldown 0)" if state.cooldown == 0 else "BLOCKED (cooldown 1)",
                fill=(224, 232, 245),
            )
            assert final.tobytes() == expected.tobytes()

    def test_semantic_labelled_is_byte_identical_to_labelled_for_all_112_states(
        self,
    ) -> None:
        recipe = build_recipe(SEMANTIC_LABELLED_VARIANT)
        assert recipe.base_render_mode == "labelled"
        assert recipe.operations == ()
        for state in arcade.all_states():
            semantic, steps = _final_image(recipe, state)
            labelled = arcade.render(state, "labelled")
            assert semantic.tobytes() == labelled.tobytes()
            assert steps[-1][1] == png_bytes(labelled)

    def test_semantic_labelled_and_labelled_have_distinct_recipe_identity(self) -> None:
        semantic = build_recipe(SEMANTIC_LABELLED_VARIANT)
        labelled = build_recipe(LABELLED_VARIANT)
        assert semantic.recipe_id != labelled.recipe_id
        state = arcade.all_states()[0]
        semantic_image, _ = _final_image(semantic, state)
        labelled_image, _ = _final_image(labelled, state)
        assert semantic_image.tobytes() == labelled_image.tobytes()

    def test_each_semantic_variant_preserves_input_ownership(self) -> None:
        for variant_id in SEMANTIC_VARIANTS:
            recipe = build_recipe(variant_id)
            base = arcade.render(READY_STATE, recipe.base_render_mode)
            before = base.tobytes()
            apply_recipe(recipe, base)
            assert base.tobytes() == before
