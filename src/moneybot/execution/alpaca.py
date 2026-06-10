"""AlpacaBroker: a thin live adapter over alpaca-py's TradingClient.

All SDK interaction is isolated in the three _*_raw methods, which return plain
dicts/lists — mirroring YFinancePriceProvider._download. Everything public is
pure translation between those primitives and moneybot's models, so tests patch
the _*_raw methods and never import the SDK or hit the network. The SDK client is
built lazily (only inside the _*_raw methods) so constructing the adapter is
free of network and import side effects.

The same TradingClient drives Alpaca's paper and live endpoints — `paper=True`
points at the paper base URL. That is what makes "paper or live by one config
flag" real. A future IBKR broker implements the same Broker Protocol; nothing
upstream changes.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from moneybot.execution.models import AccountSnapshot, Fill, PositionRecord

if TYPE_CHECKING:
    from moneybot.execution.models import OrderRequest, Side

# Our side -> Alpaca's order side. Alpaca expresses a short as a plain sell.
_ALPACA_SIDE = {"buy": "buy", "cover": "buy", "sell": "sell", "short": "sell"}

# Alpaca order status -> our Fill status.
_FILLED = {"filled"}
_REJECTED = {"rejected", "canceled", "expired"}


class AlpacaBroker:
    def __init__(
        self,
        *,
        key_id: str,
        secret_key: str,
        paper: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._key_id = key_id
        self._secret_key = secret_key
        self._paper = paper
        self._client: Any = None  # built lazily inside _get_client
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _get_client(self) -> Any:
        if self._client is None:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(
                api_key=self._key_id, secret_key=self._secret_key, paper=self._paper
            )
        return self._client

    # --- SDK boundary: the only methods that touch alpaca-py -----------------

    def _submit_raw(
        self, symbol: str, qty: int, side: str, client_order_id: str
    ) -> dict[str, Any]:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide(side),
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        order = self._get_client().submit_order(req)
        return {
            "id": str(order.id),
            "status": str(getattr(order.status, "value", order.status)),
            "filled_qty": order.filled_qty,
            "filled_avg_price": order.filled_avg_price,
        }

    def _positions_raw(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
            }
            for p in self._get_client().get_all_positions()
        ]

    def _account_raw(self) -> dict[str, Any]:
        acct = self._get_client().get_account()
        return {"equity": acct.equity, "cash": acct.cash}

    # --- Broker Protocol: pure translation -----------------------------------

    def place_order(self, order: OrderRequest) -> Fill:
        raw = self._submit_raw(
            order.ticker, order.quantity, _ALPACA_SIDE[order.side], order.client_order_id
        )
        status = raw["status"]
        if status in _FILLED:
            our_status: Side = "filled"  # type: ignore[assignment]
        elif status in _REJECTED:
            our_status = "rejected"  # type: ignore[assignment]
        else:
            our_status = "accepted"  # type: ignore[assignment]

        return Fill(
            client_order_id=order.client_order_id,
            broker_order_id=raw["id"],
            ticker=order.ticker,
            side=order.side,
            status=our_status,  # type: ignore[arg-type]
            # Phase-1 orders are whole-share, so int() truncation is safe. A partial
            # fill maps to "accepted"; the ExecutionAdapter records the position only
            # once the order reaches "filled", so partial qty here is intentionally
            # not yet persisted.
            filled_qty=int(float(raw["filled_qty"] or 0)),
            avg_price=float(raw["filled_avg_price"] or 0.0),
            ts=self._clock(),
            reason="" if status not in _REJECTED else status,
        )

    def get_positions(self) -> list[PositionRecord]:
        return [
            PositionRecord(
                ticker=p["symbol"],
                qty=float(p["qty"]),
                avg_price=float(p["avg_entry_price"]),
            )
            for p in self._positions_raw()
        ]

    def get_account(self) -> AccountSnapshot:
        raw = self._account_raw()
        # equity/cash are Optional[str] in the SDK; a funded account always has values,
        # but guard against None so an unfunded/edge account does not raise TypeError.
        return AccountSnapshot(
            equity=float(raw["equity"] or 0.0), cash=float(raw["cash"] or 0.0)
        )
