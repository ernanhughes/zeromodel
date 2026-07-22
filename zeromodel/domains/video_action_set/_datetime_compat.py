"""Python-version compatibility for the video action-set runtime."""

from __future__ import annotations

import datetime as _datetime


if not hasattr(_datetime, "UTC"):
    setattr(_datetime, "UTC", _datetime.timezone.utc)
