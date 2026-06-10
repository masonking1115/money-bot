"""The broker seam: the single contract the Execution Adapter depends on.

PaperBroker and AlpacaBroker implement it; a future IBKR broker is just one more
implementation, with no change to the adapter, store, or reconciliation. Mirrors
the LLMClient / PriceProvider Protocol seams elsewhere in moneybot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from moneybot.execution.models import (
        AccountSnapshot,
        Fill,
        OrderRequest,
        PositionRecord,
    )


@runtime_checkable
class Broker(Protocol):
    def place_order(self, order: OrderRequest) -> Fill:
        """Place one order and return the resulting Fill.

        Implementations must be idempotent on order.client_order_id: placing the
        same client_order_id twice must not result in two distinct trades.
        """
        ...

    def get_positions(self) -> list[PositionRecord]:
        """Return current non-flat positions (qty signed: long +, short -)."""
        ...

    def get_account(self) -> AccountSnapshot:
        """Return top-line account equity and cash."""
        ...
