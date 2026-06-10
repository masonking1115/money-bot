from datetime import datetime, timezone

from moneybot.config import Settings
from moneybot.execution.adapter import ExecutionAdapter
from moneybot.execution.alpaca import AlpacaBroker
from moneybot.execution.factory import build_execution_adapter
from moneybot.execution.models import OrderRequest
from moneybot.execution.paper import PaperBroker


def test_paper_mode_builds_paper_broker(tmp_path):
    settings = Settings(mode="paper", data_dir=str(tmp_path), paper_starting_cash=50_000.0)
    adapter = build_execution_adapter(settings=settings)
    assert isinstance(adapter, ExecutionAdapter)
    assert isinstance(adapter.broker, PaperBroker)
    assert adapter.broker.cash == 50_000.0
    assert adapter.store.path == tmp_path / "positions.json"


def test_live_mode_builds_alpaca_broker(tmp_path):
    settings = Settings(
        mode="live",
        data_dir=str(tmp_path),
        alpaca_key_id="k",
        alpaca_secret_key="s",
    )
    adapter = build_execution_adapter(settings=settings)
    assert isinstance(adapter.broker, AlpacaBroker)
    assert adapter.broker._client is None  # lazy: no SDK/network on construction


def test_broker_override_is_honored(tmp_path):
    settings = Settings(mode="live", data_dir=str(tmp_path))
    sentinel = PaperBroker(starting_cash=1.0)
    adapter = build_execution_adapter(settings=settings, broker=sentinel)
    assert adapter.broker is sentinel  # override wins, no Alpaca built


# H1 regression: injected clock must reach PaperBroker fills -----------------

def test_injected_clock_timestamps_fills(tmp_path):
    """H1: build_execution_adapter passes clock -> PaperBroker; fill.ts == clock time."""
    fixed_time = datetime(2025, 1, 15, 9, 30, tzinfo=timezone.utc)
    sim_clock = lambda: fixed_time

    settings = Settings(mode="paper", data_dir=str(tmp_path), paper_starting_cash=10_000.0)
    adapter = build_execution_adapter(settings=settings, clock=sim_clock)

    order = OrderRequest(
        client_order_id="test:NVDA:buy",
        ticker="NVDA",
        side="buy",
        quantity=1,
        reference_price=100.0,
    )
    fill = adapter.broker.place_order(order)

    assert fill.status == "filled"
    assert fill.ts == fixed_time
