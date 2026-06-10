"""Pure cost-basis math: how one fill mutates a position.

Single source of truth for both the paper broker's book and the position store.
qty is signed (long positive, short negative). buy/cover move qty up; sell/short
move it down. Cost basis is a weighted average while adding in the same
direction, is left unchanged while reducing, and resets to the fill price when a
position crosses through zero. Returns None when the position goes flat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moneybot.execution.models import PositionRecord

if TYPE_CHECKING:
    from moneybot.execution.models import Fill

_FLAT = 1e-9


def apply_fill(current: PositionRecord | None, fill: Fill) -> PositionRecord | None:
    signed = float(fill.filled_qty)
    if fill.side in ("sell", "short"):
        signed = -signed

    old_qty = current.qty if current is not None else 0.0
    old_avg = current.avg_price if current is not None else 0.0
    new_qty = old_qty + signed

    if abs(new_qty) < _FLAT:
        return None  # position closed out

    same_direction = old_qty == 0.0 or (old_qty > 0) == (new_qty > 0)
    if not same_direction:
        new_avg = fill.avg_price  # crossed zero: remaining shares are at the fill price
    elif abs(new_qty) > abs(old_qty):
        # adding in the same direction -> weighted-average cost
        new_avg = (abs(old_qty) * old_avg + abs(signed) * fill.avg_price) / abs(new_qty)
    else:
        new_avg = old_avg  # reducing -> cost basis unchanged

    return PositionRecord(ticker=fill.ticker, qty=new_qty, avg_price=new_avg)
