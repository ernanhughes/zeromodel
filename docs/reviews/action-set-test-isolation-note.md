# Action-set test isolation note

This stabilization change does not alter action-set production behavior.

The research instrument test patches the legacy benchmark facade with synthetic materialization and provider functions. Those patches must be scoped with `pytest.MonkeyPatch.context()` so they are restored even when an intermediate build or verification step raises.

Without exception-safe cleanup, later family-semantics and reference-verification tests observe the synthetic one-record universe and fail with misleading symptoms such as missing splice/control/temporal rows, absent provider mutation targets, and leaked lambda aliases.

The production materialization, family validation, reference verification, DTO, and SQL ownership contracts remain unchanged.
