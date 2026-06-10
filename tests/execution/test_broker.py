from datetime import datetime, timezone

from moneybot.execution.broker import Broker
from moneybot.execution.models import AccountSnapshot, Fill, OrderRequest, PositionRecord


class _StubBroker:
    def place_order(self, order: OrderRequest) -> Fill:
        return Fill(
            client_order_id=order.client_order_id,
            broker_order_id="x",
            ticker=order.ticker,
            side=order.side,
            status="filled",
            filled_qty=order.quantity,
            avg_price=1.0,
            ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
        )

    def get_positions(self) -> list[PositionRecord]:
        return []

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(equity=1.0, cash=1.0)


def test_stub_satisfies_broker_protocol():
    assert isinstance(_StubBroker(), Broker)


def test_non_broker_fails_protocol_check():
    assert not isinstance(object(), Broker)
