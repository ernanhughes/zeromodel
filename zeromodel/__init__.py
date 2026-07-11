#  zeromodel/__init__.py
"""ZeroModel package.

The package now has two layers:

- v1 experimental symbols, loaded lazily for compatibility;
- v2 artifact-kernel symbols, loaded lazily for the first-principles rebuild.

Keeping this module lightweight prevents ``import zeromodel`` or
``import zeromodel.v2`` from initializing the old runtime, global config,
DuckDB adapters, or image helpers unless those compatibility symbols are
actually requested.
"""
from __future__ import annotations

__version__ = "1.0.9"

_V1_EXPORTS = {
    "init_config": (".config", "init_config"),
    "ZeroModel": (".core", "ZeroModel"),
    "EdgeProtocol": (".edge", "EdgeProtocol"),
    "HierarchicalVPM": (".hierarchical", "HierarchicalVPM"),
    "HierarchicalEdgeProtocol": (".hierarchical_edge", "HierarchicalEdgeProtocol"),
    "DynamicNormalizer": (".normalizer", "DynamicNormalizer"),
    "AGG_MAX": (".vpm.image", "AGG_MAX"),
    "VPMImageReader": (".vpm.image", "VPMImageReader"),
    "VPMImageWriter": (".vpm.image", "VPMImageWriter"),
    "build_parent_level_png": (".vpm.image", "build_parent_level_png"),
    "get_critical_tile": (".vpm.transform", "get_critical_tile"),
    "transform_vpm": (".vpm.transform", "transform_vpm"),
}

_V2_EXPORTS = {
    "LayoutRecipe": (".v2", "LayoutRecipe"),
    "ScoreTable": (".v2", "ScoreTable"),
    "VPMArtifact": (".v2", "VPMArtifact"),
    "VPMCell": (".v2", "VPMCell"),
    "VPMRegion": (".v2", "VPMRegion"),
    "build_vpm": (".v2", "build_vpm"),
}

_EXPORTS = {**_V1_EXPORTS, **_V2_EXPORTS}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError("module 'zeromodel' has no attribute %r" % name)

    module_name, attribute_name = _EXPORTS[name]
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals()) + __all__)
