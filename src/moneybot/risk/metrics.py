"""Pure price/volume math for the Risk Engine.

No I/O and no pandas — callers pass plain sequences so these are trivially
unit-testable (mirrors analyst/relative_strength.py). Returns None when a metric
is not computable so the engine can veto on missing data rather than guess.
"""

from __future__ import annotations

from collections.abc import Sequence


def realized_volatility(closes: Sequence[float | None]) -> float | None:
    """Sample standard deviation of period-over-period simple returns.

    None if fewer than three valid closes (need >=2 returns for a sample stddev).
    A flat series returns 0.0.
    """
    vals = [c for c in closes if c is not None]
    if len(vals) < 3:
        return None
    returns = [(vals[i] / vals[i - 1]) - 1.0 for i in range(1, len(vals)) if vals[i - 1] != 0]
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return var**0.5


def average_dollar_volume(
    closes: Sequence[float | None],
    volumes: Sequence[float | None],
) -> float | None:
    """Mean of close*volume over bars where both are present. None if no such bar."""
    pairs = [(c, v) for c, v in zip(closes, volumes) if c is not None and v is not None]
    if not pairs:
        return None
    return sum(c * v for c, v in pairs) / len(pairs)
