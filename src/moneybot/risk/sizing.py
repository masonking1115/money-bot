"""Pure position-sizing math for the Risk Engine.

Conviction sets the base slice (capped at max_position_pct); volatility-scaling
then trims names more volatile than the target so no single name dominates
portfolio risk. Scaling never increases a position past its conviction-capped
base (we trim risk, we never lever up a calm name). Output is a fraction of
equity in [0, max_position_pct].
"""

from __future__ import annotations


def target_weight(
    *,
    conviction: float,
    volatility: float | None,
    max_position_pct: float,
    target_volatility: float,
) -> float:
    """Fraction of equity to allocate to one name."""
    base = conviction * max_position_pct
    if volatility is not None and volatility > 0:
        scale = min(1.0, target_volatility / volatility)
        base *= scale
    return max(0.0, min(base, max_position_pct))
