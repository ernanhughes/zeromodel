from __future__ import annotations

from enum import Enum


class ScoreSemantics(str, Enum):
    """How a dimension's raw score should be interpreted - never inferred
    from its name or family. A report adapter declares this explicitly per
    dimension so that "0.9" means the same measurable thing regardless of
    whether the dimension happens to be named in a way that sounds good or
    bad."""

    HIGHER_IS_BETTER = "higher_is_better"
    HIGHER_IS_WORSE = "higher_is_worse"
    TARGET_RANGE = "target_range"
    DESCRIPTIVE = "descriptive"
