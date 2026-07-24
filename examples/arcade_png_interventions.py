#!/usr/bin/env python3
"""Deterministic PNG intervention recipes for the controlled representation benchmark.

Every recipe describes a declared, content-addressed sequence of *pure* pixel
operations applied to one canonical arcade render (see
`examples/local_model_zero_arcade_test.py:render`). Recipes never touch the
provider, the policy, the fixture state universe, or the prompt - they only
change what the rendered PNG looks like. See
`docs/research/controlled-png-representation-benchmark.md` for the research
question this supports.

Intervention functions are pure: `(image, canonical_parameters) -> image`.
They never mutate their input `Image.Image` in place, and the cooldown
operations *read* the ready/blocked state from the existing colour-coded
indicator already baked into the base render's pixels rather than accepting
it as a side-channel parameter - so a recipe's declared parameters stay
identical across every fixture state, and recipe identity therefore depends
only on the declared transformation, not on which state it will later be
applied to.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from math import hypot

from PIL import Image, ImageDraw
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256

RECIPE_VERSION = "arcade-png-intervention-recipe/v1"

# Geometry mirrors `local_model_zero_arcade_test.render()` exactly so overlays
# align with the sprites and lane boundaries that function already drew. Kept
# as local constants rather than imported from that module so this module has
# no dependency on it - only pure PIL pixel operations live here.
IMG_WIDTH = 896
IMG_HEIGHT = 512
LANES = 7
LEFT_MARGIN = 42
RIGHT_MARGIN = 42
HUD_HEIGHT = 72
UNLABELLED_BOTTOM = IMG_HEIGHT - 24

BG_COLOR = (13, 18, 30)
GRID_COLOR = (53, 66, 92)
WHITE = (224, 232, 245)
GREEN = (70, 225, 125)
RED = (242, 74, 74)

PRIMARY_COOLDOWN_BOX = (20, 18, 56, 54)
SECONDARY_COOLDOWN_BOX = (IMG_WIDTH - 56, 18, IMG_WIDTH - 20, 54)

# The tank sprite's lowest point is `bottom - 45 + 33` = `488 - 45 + 33` = 476
# for every lane and every fixture state (see `render()`'s wheel polygon) -
# `FOOTER_TOP` stays below that with a 4px margin so the footer band never
# occludes the sprite it sits under.
FOOTER_TOP = IMG_HEIGHT - 32
FOOTER_BG = (19, 26, 42)
FOOTER_BORDER = WHITE
FOOTER_TEXT_Y = FOOTER_TOP + 4
COOLDOWN_TEXT_POSITION = (68, 27)


@dataclass(frozen=True, slots=True)
class ArcadePngOperationSpec:
    """One declared step in an intervention recipe."""

    operation: str
    operation_version: str
    parameters: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "operation_version": self.operation_version,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class ArcadePngInterventionRecipe:
    """An immutable, content-addressed PNG representation recipe.

    ``recipe_id`` is derived from ``canonical_sha256`` over every other field,
    exactly like every other identity in the video action-set domain (see
    ``ProviderConfigurationDTO.provider_configuration_id``). ``operations`` may
    be empty for the two reference variants (``labelled-v1``/``unlabelled-v1``),
    which declare no PNG intervention beyond the canonical base render itself.
    """

    version: str
    variant_id: str
    base_render_mode: str
    operations: tuple[ArcadePngOperationSpec, ...]
    metadata: Mapping[str, object]
    recipe_id: str

    def __post_init__(self) -> None:
        if self.version != RECIPE_VERSION:
            raise ValueError("unsupported arcade png intervention recipe version")
        if not self.variant_id:
            raise ValueError("recipe variant_id must be non-empty")
        if self.base_render_mode not in ("labelled", "unlabelled"):
            raise ValueError("recipe base_render_mode must be labelled or unlabelled")
        expected_id = canonical_sha256(_recipe_payload_without_id(self))
        if self.recipe_id != expected_id:
            raise ValueError("recipe id mismatch")

    @classmethod
    def build(
        cls,
        *,
        variant_id: str,
        base_render_mode: str,
        operations: Sequence[ArcadePngOperationSpec],
        metadata: Mapping[str, object] | None = None,
    ) -> ArcadePngInterventionRecipe:
        payload = {
            "version": RECIPE_VERSION,
            "variant_id": variant_id,
            "base_render_mode": base_render_mode,
            "operations": [op.to_dict() for op in operations],
            "metadata": dict(metadata or {}),
        }
        recipe_id = canonical_sha256(payload)
        return cls(
            version=RECIPE_VERSION,
            variant_id=variant_id,
            base_render_mode=base_render_mode,
            operations=tuple(operations),
            metadata=dict(metadata or {}),
            recipe_id=recipe_id,
        )

    def to_dict(self) -> dict[str, object]:
        return _recipe_payload_without_id(self) | {"recipe_id": self.recipe_id}


def _recipe_payload_without_id(
    recipe: ArcadePngInterventionRecipe,
) -> dict[str, object]:
    return {
        "version": recipe.version,
        "variant_id": recipe.variant_id,
        "base_render_mode": recipe.base_render_mode,
        "operations": [op.to_dict() for op in recipe.operations],
        "metadata": dict(recipe.metadata),
    }


def png_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def _distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _detect_cooldown_from_pixels(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    ready_color: tuple[int, int, int],
    blocked_color: tuple[int, int, int],
    tolerance: float = 60.0,
) -> int:
    """Read the ready/blocked state directly from the existing colour-coded
    indicator pixels rather than being told it out-of-band. Keeps every
    cooldown intervention a pure function of (image, declared parameters)."""
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    pixel = image.convert("RGB").getpixel((cx, cy))
    if _distance(pixel, ready_color) <= tolerance:
        return 0
    if _distance(pixel, blocked_color) <= tolerance:
        return 1
    raise ValueError(f"cannot detect cooldown indicator colour at {box}: pixel={pixel}")


def _draw_ready_shape(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    outline: tuple[int, int, int],
    fill: tuple[int, int, int] | None,
    width: int,
) -> None:
    draw.ellipse(box, outline=outline, fill=fill, width=width)


def _draw_blocked_shape(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    outline: tuple[int, int, int],
    fill: tuple[int, int, int] | None,
    width: int,
) -> None:
    if fill is not None:
        draw.rectangle(box, fill=fill)
    x0, y0, x1, y1 = box
    draw.line((x0, y0, x1, y1), fill=outline, width=width)
    draw.line((x0, y1, x1, y0), fill=outline, width=width)
    draw.rectangle(box, outline=outline, width=2)


def _op_cooldown_shape_overlay(
    image: Image.Image, parameters: Mapping[str, object]
) -> Image.Image:
    """Replace the coloured cooldown indicator with a shape (circle=ready,
    cross=blocked) so the state is distinguishable in grayscale, without text."""
    box = tuple(parameters["position"])  # type: ignore[arg-type]
    ready_color = tuple(parameters["ready_color"])  # type: ignore[arg-type]
    blocked_color = tuple(parameters["blocked_color"])  # type: ignore[arg-type]
    background = tuple(parameters["background_color"])  # type: ignore[arg-type]
    outline = tuple(parameters["outline_color"])  # type: ignore[arg-type]
    cooldown = _detect_cooldown_from_pixels(
        image, box, ready_color=ready_color, blocked_color=blocked_color
    )
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle(box, fill=background)
    if cooldown == 0:
        _draw_ready_shape(draw, box, outline=outline, fill=None, width=4)
    else:
        _draw_blocked_shape(draw, box, outline=outline, fill=None, width=4)
    return result


def _op_cooldown_dual_overlay(
    image: Image.Image, parameters: Mapping[str, object]
) -> Image.Image:
    """Cooldown via shape AND colour (green circle=ready, red cross=blocked)."""
    box = tuple(parameters["position"])  # type: ignore[arg-type]
    ready_color = tuple(parameters["ready_color"])  # type: ignore[arg-type]
    blocked_color = tuple(parameters["blocked_color"])  # type: ignore[arg-type]
    background = tuple(parameters["background_color"])  # type: ignore[arg-type]
    outline = tuple(parameters["outline_color"])  # type: ignore[arg-type]
    cooldown = _detect_cooldown_from_pixels(
        image, box, ready_color=ready_color, blocked_color=blocked_color
    )
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle(box, fill=background)
    fill = ready_color if cooldown == 0 else blocked_color
    if cooldown == 0:
        _draw_ready_shape(draw, box, outline=outline, fill=fill, width=4)
    else:
        _draw_blocked_shape(draw, box, outline=outline, fill=fill, width=4)
    return result


def _op_cooldown_marker_duplicate(
    image: Image.Image, parameters: Mapping[str, object]
) -> Image.Image:
    """Duplicate an already-drawn cooldown marker at a second declared
    location, by copying pixels rather than re-detecting state - this makes
    the duplication itself the recorded operation, matching the source glyph
    exactly regardless of detection tolerance."""
    source_box = tuple(parameters["source_position"])  # type: ignore[arg-type]
    target_box = tuple(parameters["target_position"])  # type: ignore[arg-type]
    result = image.copy()
    glyph = image.crop(source_box)
    result.paste(glyph, (target_box[0], target_box[1]))
    return result


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    radius: float,
    shape: str,
    color: tuple[int, int, int],
) -> None:
    if shape == "triangle":
        draw.polygon(
            [(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)],
            fill=color,
        )
    else:
        draw.polygon(
            [
                (cx, cy - radius),
                (cx + radius, cy),
                (cx, cy + radius),
                (cx - radius, cy),
            ],
            fill=color,
        )


def _op_lane_separator_enhance(
    image: Image.Image, parameters: Mapping[str, object]
) -> Image.Image:
    """Strengthen lane-position evidence: brighter/thicker separators plus
    alternating triangle/diamond markers repeated above and below the lane
    band at every boundary. Never touches the cooldown indicator or draws
    numeric lane labels."""
    left = int(parameters["left"])  # type: ignore[arg-type]
    right = int(parameters["right"])  # type: ignore[arg-type]
    hud = int(parameters["hud"])  # type: ignore[arg-type]
    bottom = int(parameters["bottom"])  # type: ignore[arg-type]
    width = int(parameters["width"])  # type: ignore[arg-type]
    lanes = int(parameters["lanes"])  # type: ignore[arg-type]
    color = tuple(parameters["separator_color"])  # type: ignore[arg-type]
    line_width = int(parameters["separator_width"])  # type: ignore[arg-type]
    marker_radius = int(parameters["marker_radius"])  # type: ignore[arg-type]

    result = image.copy()
    draw = ImageDraw.Draw(result)
    lane_w = (width - left - right) / lanes
    for index in range(lanes + 1):
        x = round(left + index * lane_w)
        draw.line((x, hud, x, bottom), fill=color, width=line_width)
        shape = "triangle" if index % 2 == 0 else "diamond"
        # `- 4` (not the larger bottom-side offset) keeps the top marker
        # below y=54 so it never overlaps `PRIMARY_COOLDOWN_BOX`
        # (y in [18, 54]) even at the leftmost boundary (x == left margin).
        _draw_marker(draw, x, hud - marker_radius - 4, marker_radius, shape, color)
        _draw_marker(draw, x, bottom + marker_radius + 6, marker_radius, shape, color)
    return result


def _op_footer_reserved_area(
    image: Image.Image, parameters: Mapping[str, object]
) -> Image.Image:
    """Introduce a single structured reserved footer band: a distinct fill
    colour plus one top border line. Draws no numerals, no cooldown text, and
    no per-lane boundary ticks - isolates whether reserved geometry alone
    (with no semantic content in it) changes provider behaviour."""
    top = int(parameters["top"])  # type: ignore[arg-type]
    width = int(parameters["width"])  # type: ignore[arg-type]
    height = int(parameters["height"])  # type: ignore[arg-type]
    fill = tuple(parameters["fill_color"])  # type: ignore[arg-type]
    border = tuple(parameters["border_color"])  # type: ignore[arg-type]
    border_width = int(parameters["border_width"])  # type: ignore[arg-type]
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, top, width, height), fill=fill)
    draw.line((0, top, width, top), fill=border, width=border_width)
    return result


def _op_lane_numerals_overlay(
    image: Image.Image, parameters: Mapping[str, object]
) -> Image.Image:
    """Draw the numeral 0..6 centred under each of the seven lane *regions*.

    Deliberately iterates ``range(lanes)`` (seven centres), never
    ``range(lanes + 1)`` (eight boundaries) - the Stage 2E
    ``lane-enhanced-v1`` real-provider result over-emphasized the eight
    boundaries between lanes and regressed (one rejected response, more
    action-changing errors); this operation encodes lane position directly
    at each region's centre instead."""
    left = int(parameters["left"])  # type: ignore[arg-type]
    right = int(parameters["right"])  # type: ignore[arg-type]
    width = int(parameters["width"])  # type: ignore[arg-type]
    lanes = int(parameters["lanes"])  # type: ignore[arg-type]
    text_y = int(parameters["text_y"])  # type: ignore[arg-type]
    color = tuple(parameters["text_color"])  # type: ignore[arg-type]
    result = image.copy()
    draw = ImageDraw.Draw(result)
    lane_w = (width - left - right) / lanes
    for index in range(lanes):
        cx = left + (index + 0.5) * lane_w
        draw.text((cx - 4, text_y), str(index), fill=color)
    return result


def _op_cooldown_text_overlay(
    image: Image.Image, parameters: Mapping[str, object]
) -> Image.Image:
    """Draw an explicit READY/BLOCKED text label next to the existing
    colour-coded cooldown indicator, reading state from its pixels exactly
    like the shape/dual cooldown operations do."""
    box = tuple(parameters["position"])  # type: ignore[arg-type]
    ready_color = tuple(parameters["ready_color"])  # type: ignore[arg-type]
    blocked_color = tuple(parameters["blocked_color"])  # type: ignore[arg-type]
    text_position = tuple(parameters["text_position"])  # type: ignore[arg-type]
    color = tuple(parameters["text_color"])  # type: ignore[arg-type]
    cooldown = _detect_cooldown_from_pixels(
        image, box, ready_color=ready_color, blocked_color=blocked_color
    )
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.text(text_position, "READY" if cooldown == 0 else "BLOCKED", fill=color)
    return result


OPERATION_FUNCTIONS: dict[
    str, Callable[[Image.Image, Mapping[str, object]], Image.Image]
] = {
    "cooldown_shape_overlay": _op_cooldown_shape_overlay,
    "cooldown_dual_overlay": _op_cooldown_dual_overlay,
    "cooldown_marker_duplicate": _op_cooldown_marker_duplicate,
    "lane_separator_enhance": _op_lane_separator_enhance,
    "footer_reserved_area": _op_footer_reserved_area,
    "lane_numerals_overlay": _op_lane_numerals_overlay,
    "cooldown_text_overlay": _op_cooldown_text_overlay,
}


LABELLED_VARIANT = "labelled-v1"
UNLABELLED_VARIANT = "unlabelled-v1"
REFERENCE_VARIANTS = (LABELLED_VARIANT, UNLABELLED_VARIANT)

COOLDOWN_SHAPE_VARIANT = "cooldown-shape-v1"
COOLDOWN_DUAL_VARIANT = "cooldown-dual-v1"
COOLDOWN_REDUNDANT_VARIANT = "cooldown-redundant-v1"
COOLDOWN_VARIANTS = (
    COOLDOWN_SHAPE_VARIANT,
    COOLDOWN_DUAL_VARIANT,
    COOLDOWN_REDUNDANT_VARIANT,
)

LANE_ENHANCED_VARIANT = "lane-enhanced-v1"
LANE_VARIANTS = (LANE_ENHANCED_VARIANT,)

FOOTER_ONLY_VARIANT = "footer-only-v1"
LANE_NUMERALS_VARIANT = "lane-numerals-v1"
COOLDOWN_TEXT_VARIANT = "cooldown-text-v1"
SEMANTIC_LABELLED_VARIANT = "semantic-labelled-v1"
SEMANTIC_VARIANTS = (
    FOOTER_ONLY_VARIANT,
    LANE_NUMERALS_VARIANT,
    COOLDOWN_TEXT_VARIANT,
    SEMANTIC_LABELLED_VARIANT,
)

COMBINED_VARIANT = "combined-v1"

ALL_VARIANTS = (
    REFERENCE_VARIANTS
    + COOLDOWN_VARIANTS
    + LANE_VARIANTS
    + SEMANTIC_VARIANTS
    + (COMBINED_VARIANT,)
)


def _cooldown_shape_ops() -> tuple[ArcadePngOperationSpec, ...]:
    return (
        ArcadePngOperationSpec(
            operation="cooldown_shape_overlay",
            operation_version="v1",
            parameters={
                "position": list(PRIMARY_COOLDOWN_BOX),
                "ready_color": list(GREEN),
                "blocked_color": list(RED),
                "background_color": list(BG_COLOR),
                "outline_color": list(WHITE),
            },
        ),
    )


def _cooldown_dual_ops() -> tuple[ArcadePngOperationSpec, ...]:
    return (
        ArcadePngOperationSpec(
            operation="cooldown_dual_overlay",
            operation_version="v1",
            parameters={
                "position": list(PRIMARY_COOLDOWN_BOX),
                "ready_color": list(GREEN),
                "blocked_color": list(RED),
                "background_color": list(BG_COLOR),
                "outline_color": list(WHITE),
            },
        ),
    )


def _cooldown_redundant_ops() -> tuple[ArcadePngOperationSpec, ...]:
    return _cooldown_dual_ops() + (
        ArcadePngOperationSpec(
            operation="cooldown_marker_duplicate",
            operation_version="v1",
            parameters={
                "source_position": list(PRIMARY_COOLDOWN_BOX),
                "target_position": list(SECONDARY_COOLDOWN_BOX),
            },
        ),
    )


def _lane_enhanced_ops() -> tuple[ArcadePngOperationSpec, ...]:
    return (
        ArcadePngOperationSpec(
            operation="lane_separator_enhance",
            operation_version="v1",
            parameters={
                "left": LEFT_MARGIN,
                "right": RIGHT_MARGIN,
                "hud": HUD_HEIGHT,
                "bottom": UNLABELLED_BOTTOM,
                "width": IMG_WIDTH,
                "lanes": LANES,
                "separator_color": list(WHITE),
                "separator_width": 4,
                "marker_radius": 6,
            },
        ),
    )


def _footer_reserved_area_ops() -> tuple[ArcadePngOperationSpec, ...]:
    return (
        ArcadePngOperationSpec(
            operation="footer_reserved_area",
            operation_version="v1",
            parameters={
                "top": FOOTER_TOP,
                "width": IMG_WIDTH,
                "height": IMG_HEIGHT,
                "fill_color": list(FOOTER_BG),
                "border_color": list(FOOTER_BORDER),
                "border_width": 2,
            },
        ),
    )


def _lane_numerals_ops() -> tuple[ArcadePngOperationSpec, ...]:
    # The footer must be applied first: lane numerals are drawn *within* the
    # reserved footer band it introduces, not as a standalone overlay.
    return _footer_reserved_area_ops() + (
        ArcadePngOperationSpec(
            operation="lane_numerals_overlay",
            operation_version="v1",
            parameters={
                "left": LEFT_MARGIN,
                "right": RIGHT_MARGIN,
                "width": IMG_WIDTH,
                "lanes": LANES,
                "text_y": FOOTER_TEXT_Y,
                "text_color": list(WHITE),
            },
        ),
    )


def _cooldown_text_ops() -> tuple[ArcadePngOperationSpec, ...]:
    return (
        ArcadePngOperationSpec(
            operation="cooldown_text_overlay",
            operation_version="v1",
            parameters={
                "position": list(PRIMARY_COOLDOWN_BOX),
                "ready_color": list(GREEN),
                "blocked_color": list(RED),
                "text_position": list(COOLDOWN_TEXT_POSITION),
                "text_color": list(WHITE),
            },
        ),
    )


def _semantic_labelled_ops() -> tuple[ArcadePngOperationSpec, ...]:
    return _lane_numerals_ops() + _cooldown_text_ops()


_VARIANT_OPERATION_BUILDERS: dict[
    str, Callable[[], tuple[ArcadePngOperationSpec, ...]]
] = {
    COOLDOWN_SHAPE_VARIANT: _cooldown_shape_ops,
    COOLDOWN_DUAL_VARIANT: _cooldown_dual_ops,
    COOLDOWN_REDUNDANT_VARIANT: _cooldown_redundant_ops,
    LANE_ENHANCED_VARIANT: _lane_enhanced_ops,
    FOOTER_ONLY_VARIANT: _footer_reserved_area_ops,
    LANE_NUMERALS_VARIANT: _lane_numerals_ops,
    COOLDOWN_TEXT_VARIANT: _cooldown_text_ops,
    SEMANTIC_LABELLED_VARIANT: _semantic_labelled_ops,
}


def build_recipe(
    variant_id: str,
    *,
    combined_cooldown: str | None = None,
    combined_lane: str | None = None,
) -> ArcadePngInterventionRecipe:
    """Build the declared, content-addressed recipe for one variant id.

    ``combined_cooldown``/``combined_lane`` are required (and must each name a
    variant from the matching family) only when ``variant_id == "combined-v1"``
    - never hard-code a winner here.
    """
    if variant_id not in ALL_VARIANTS:
        raise ValueError(f"unknown representation variant: {variant_id!r}")
    if variant_id == LABELLED_VARIANT:
        return ArcadePngInterventionRecipe.build(
            variant_id=variant_id,
            base_render_mode="labelled",
            operations=(),
            metadata={"family": "reference"},
        )
    if variant_id == UNLABELLED_VARIANT:
        return ArcadePngInterventionRecipe.build(
            variant_id=variant_id,
            base_render_mode="unlabelled",
            operations=(),
            metadata={"family": "reference"},
        )
    if variant_id in _VARIANT_OPERATION_BUILDERS:
        if variant_id in COOLDOWN_VARIANTS:
            family = "cooldown"
        elif variant_id in LANE_VARIANTS:
            family = "lane"
        else:
            family = "semantic"
        return ArcadePngInterventionRecipe.build(
            variant_id=variant_id,
            base_render_mode="unlabelled",
            operations=_VARIANT_OPERATION_BUILDERS[variant_id](),
            metadata={"family": family},
        )
    if combined_cooldown not in COOLDOWN_VARIANTS:
        raise ValueError("--combined-cooldown must name a cooldown-family variant")
    if combined_lane not in LANE_VARIANTS:
        raise ValueError("--combined-lane must name a lane-family variant")
    ops = (
        _VARIANT_OPERATION_BUILDERS[combined_cooldown]()
        + _VARIANT_OPERATION_BUILDERS[combined_lane]()
    )
    return ArcadePngInterventionRecipe.build(
        variant_id=variant_id,
        base_render_mode="unlabelled",
        operations=ops,
        metadata={
            "family": "combined",
            "combined_cooldown": combined_cooldown,
            "combined_lane": combined_lane,
        },
    )


def apply_recipe(
    recipe: ArcadePngInterventionRecipe, base_image: Image.Image
) -> list[tuple[Image.Image, bytes]]:
    """Apply every declared operation in order, returning
    ``[(base_image, base_bytes), (after_op_1, bytes_1), ...]``.

    Raises ``ValueError`` if any declared operation produces byte-identical
    PNG output to its input - a no-op transform is never accepted silently.
    """
    steps: list[tuple[Image.Image, bytes]] = [(base_image, png_bytes(base_image))]
    current_image = base_image
    for spec in recipe.operations:
        fn = OPERATION_FUNCTIONS.get(spec.operation)
        if fn is None:
            raise ValueError(f"unknown intervention operation: {spec.operation!r}")
        next_image = fn(current_image, spec.parameters)
        next_bytes = png_bytes(next_image)
        if next_bytes == steps[-1][1]:
            raise ValueError(
                f"intervention operation {spec.operation!r} produced no pixel change"
            )
        steps.append((next_image, next_bytes))
        current_image = next_image
    return steps


__all__ = [
    "ALL_VARIANTS",
    "COMBINED_VARIANT",
    "COOLDOWN_DUAL_VARIANT",
    "COOLDOWN_REDUNDANT_VARIANT",
    "COOLDOWN_SHAPE_VARIANT",
    "COOLDOWN_TEXT_POSITION",
    "COOLDOWN_TEXT_VARIANT",
    "COOLDOWN_VARIANTS",
    "FOOTER_ONLY_VARIANT",
    "FOOTER_TOP",
    "IMG_HEIGHT",
    "IMG_WIDTH",
    "LABELLED_VARIANT",
    "LANES",
    "LANE_ENHANCED_VARIANT",
    "LANE_NUMERALS_VARIANT",
    "LANE_VARIANTS",
    "LEFT_MARGIN",
    "OPERATION_FUNCTIONS",
    "PRIMARY_COOLDOWN_BOX",
    "RECIPE_VERSION",
    "REFERENCE_VARIANTS",
    "RIGHT_MARGIN",
    "SECONDARY_COOLDOWN_BOX",
    "SEMANTIC_LABELLED_VARIANT",
    "SEMANTIC_VARIANTS",
    "UNLABELLED_VARIANT",
    "ArcadePngInterventionRecipe",
    "ArcadePngOperationSpec",
    "apply_recipe",
    "build_recipe",
    "png_bytes",
]
