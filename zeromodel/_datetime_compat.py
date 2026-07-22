"""Compatibility helpers for the supported Python 3.10+ runtime range."""
from __future__ import annotations

import datetime as _datetime


def ensure_datetime_utc() -> None:
    """Expose ``datetime.UTC`` on Python 3.10 before domain modules import it."""
    if not hasattr(_datetime, "UTC"):
        setattr(_datetime, "UTC", _datetime.timezone.utc)


ensure_datetime_utc()

__all__ = ["ensure_datetime_utc"]
