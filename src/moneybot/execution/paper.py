"""PaperBroker: a deterministic in-memory simulated broker (no network, no LLM).

Fills market orders instantly at the order's reference_price (rejecting if none
is supplied). Tracks cash and positions internally. Equity is marked-to-cost
(cash + sum(qty * avg_price)) — the orchestrator re-marks to market with live
prices when it builds a PortfolioState. Idempotent on client_order_id: a repeated
order returns the original fill without re-applying it. This is the validated
default path; the same code runs in backtests and paper trading.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from moneybot.execution.models import AccountSnapshot, Fill, PositionRecord
from moneybot.execution.positions import apply_fill

if TYPE_CHECKING:
    from moneybot.execution.models import OrderRequest


class PaperBroker:
    def __init__(
        self,
        *,
        starting_cash: float,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.cash = starting_cash
        self._positions: dict[str, PositionRecord] = {}
        self._fills: dict[str, Fill] = {}  # client_order_id -> Fill (idempotency)
        self._seq = 0
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def place_order(self, order: OrderRequest) -> Fill:
        prior = self._fills.get(order.client_order_id)
        if prior is not None:
            return prior  # idempotent: do not re-apply

        price = order.reference_price
        if price is None or price <= 0:
            return Fill(
                client_order_id=order.client_order_id,
                broker_order_id="paper-rejected",
                ticker=order.ticker,
                side=order.side,
                status="rejected",
                ts=self._clock(),
                reason="no reference price to fill against",
            )

        self._seq += 1
        fill = Fill(
            client_order_id=order.client_order_id,
            broker_order_id=f"paper-{self._seq}",
            ticker=order.ticker,
            side=order.side,
            status="filled",
            filled_qty=order.quantity,
            avg_price=price,
            ts=self._clock(),
        )

        # buy/cover cost cash; sell/short add cash.
        signed = order.quantity if order.side in ("buy", "cover") else -order.quantity
        self.cash -= signed * price

        updated = apply_fill(self._positions.get(order.ticker), fill)
        if updated is None:
            self._positions.pop(order.ticker, None)
        else:
            self._positions[order.ticker] = updated

        self._fills[order.client_order_id] = fill
        return fill

    def get_positions(self) -> list[PositionRecord]:
        return list(self._positions.values())

    def get_account(self) -> AccountSnapshot:
        marked = sum(p.qty * p.avg_price for p in self._positions.values())
        return AccountSnapshot(equity=self.cash + marked, cash=self.cash)
