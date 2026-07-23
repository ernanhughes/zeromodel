"""Public release-candidate validator entry point.

The full implementation is kept in ``_validate_release_candidate_impl.py`` so
this compatibility entry point can apply small, independently tested verdict
hardening without duplicating or rewriting the release/build machinery.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_IMPL_PATH = Path(__file__).with_name("_validate_release_candidate_impl.py")
_SPEC = importlib.util.spec_from_file_location(
    "_zeromodel_validate_release_candidate_impl", _IMPL_PATH
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery
    raise RuntimeError(f"cannot load release validator implementation: {_IMPL_PATH}")
_impl = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _impl
_SPEC.loader.exec_module(_impl)


def _evaluate_required_layer(
    name: str, counts: Mapping[str, Any] | None
):
    """Require a required layer to execute at least one test.

    Collection alone is not execution: a layer whose entire selected set is
    skipped must fail the release verdict just like a missing or zero-collected
    layer. This closes the remaining false-positive path after the initial
    release-truth stabilization.
    """
    if counts is None:
        return _impl.ReleaseLayerVerdict(
            name,
            _impl._LAYER_STATUS_NOT_EXECUTED,
            ("required layer result is missing",),
        )

    reasons: list[str] = []
    returncode = counts.get("returncode")
    if returncode != 0:
        reasons.append(f"returncode={returncode!r}")

    passed = int(counts.get("passed") or 0)
    failed = int(counts.get("failed") or 0)
    errors = int(counts.get("errors") or 0)
    skipped = int(counts.get("skipped") or 0)

    if failed:
        reasons.append(f"failed={failed}")
    if errors:
        reasons.append(f"errors={errors}")

    collected = passed + failed + errors + skipped
    executed = passed + failed + errors
    if collected == 0:
        reasons.append("collected zero relevant tests")
    elif executed == 0:
        reasons.append("executed zero required tests; all collected tests were skipped")

    if reasons:
        return _impl.ReleaseLayerVerdict(
            name, _impl._LAYER_STATUS_FAILED, tuple(reasons)
        )
    return _impl.ReleaseLayerVerdict(name, _impl._LAYER_STATUS_PASSED)


# Functions defined in the implementation resolve globals in that module, so
# patch its verdict primitive before re-exporting the established API.
_impl._evaluate_required_layer = _evaluate_required_layer

for _name, _value in vars(_impl).items():
    if _name in {
        "__name__",
        "__file__",
        "__package__",
        "__loader__",
        "__spec__",
        "_evaluate_required_layer",
    }:
        continue
    globals()[_name] = _value


if __name__ == "__main__":
    raise SystemExit(_impl.main())
