"""Pure reconciliation: compare the bot's stored positions against the broker's.

Report-only. It never places orders to "fix" drift — surfacing a discrepancy is
a safety event for the operator/orchestrator to handle, not something to auto-
trade away. A position present on only one side reports the other side as 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moneybot.execution.models import Discrepancy, ReconciliationResult

if TYPE_CHECKING:
    from moneybot.execution.models import PositionRecord

_TOLERANCE = 1e-6


def reconcile(
    stored: list[PositionRecord],
    broker: list[PositionRecord],
) -> ReconciliationResult:
    stored_qty = {p.ticker: p.qty for p in stored}
    broker_qty = {p.ticker: p.qty for p in broker}

    discrepancies: list[Discrepancy] = []
    for ticker in sorted(set(stored_qty) | set(broker_qty)):
        s = stored_qty.get(ticker, 0.0)
        b = broker_qty.get(ticker, 0.0)
        if abs(s - b) > _TOLERANCE:
            discrepancies.append(Discrepancy(ticker=ticker, stored_qty=s, broker_qty=b))

    return ReconciliationResult(in_sync=not discrepancies, discrepancies=discrepancies)
