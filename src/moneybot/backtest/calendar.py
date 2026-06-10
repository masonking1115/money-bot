"""Derive the replay trading calendar from the benchmark's real bar dates."""

from __future__ import annotations

from datetime import date

import pandas as pd


def trading_days_from_bars(bars: pd.DataFrame, start: date, end: date) -> list[date]:
    """Sorted, de-duplicated dates from `bars['ts']` that fall within [start, end]."""
    if bars.empty:
        return []
    days = sorted({ts.date() for ts in pd.to_datetime(bars["ts"])})
    return [d for d in days if start <= d <= end]
