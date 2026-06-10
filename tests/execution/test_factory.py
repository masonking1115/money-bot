from moneybot.config import Settings
from moneybot.execution.adapter import ExecutionAdapter
from moneybot.execution.alpaca import AlpacaBroker
from moneybot.execution.factory import build_execution_adapter
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
