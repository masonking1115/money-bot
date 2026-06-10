"""Pure mechanical-exit evaluation for open long positions.

Given open positions, their entry dates, current prices, and the strategy's
ExitPlan, decide which longs to close and why. Precedence: stop-loss (capital
preservation) first, then profit-target, then time-stop. A position with no
current price is skipped (cannot evaluate); a position with no known entry date
can still stop/target but not time-stop. Phase-1 manages longs only — short
(hedge) lifecycle is deferred.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moneybot.orchestrator.models import ExitSignal

if TYPE_CHECKING:
    from datetime import date

    from moneybot.execution.models import PositionRecord
    from moneybot.strategies.models import ExitPlan


def evaluate_exits(
    *,
    positions: list[PositionRecord],
    entry_dates: dict[str, date],
    current_prices: dict[str, float],
    exit_plan: ExitPlan,
    as_of: date,
) -> list[ExitSignal]:
    signals: list[ExitSignal] = []
    for pos in positions:
        shares = int(pos.qty)
        if shares <= 0:
            # Skip shorts (qty < 0) and sub-share fractional longs (0 < qty < 1) —
            # an ExitSignal is whole-share. This bot's own orders are whole-share, so
            # a fractional long would only arise from an externally-placed position.
            continue
        price = current_prices.get(pos.ticker)
        if price is None:
            continue

        entry = pos.avg_price
        reason: str | None = None
        if price <= entry * (1 - exit_plan.stop_loss_pct):
            reason = "stop_loss"
        elif price >= entry * (1 + exit_plan.profit_target_pct):
            reason = "profit_target"
        else:
            entry_date = entry_dates.get(pos.ticker)
            if entry_date is not None and (as_of - entry_date).days >= exit_plan.max_hold_days:
                reason = "time_stop"

        if reason is not None:
            signals.append(
                ExitSignal(
                    ticker=pos.ticker, shares=shares, reason=reason, reference_price=price
                )
            )
    return signals
