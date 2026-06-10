from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from moneybot.config import TickerMeta, Universe
from moneybot.execution.models import AccountSnapshot, Fill, PositionRecord
from moneybot.memory.models import JournalEntry
from moneybot.orchestrator.engine import Orchestrator
from moneybot.strategies.models import ExitPlan

ET = ZoneInfo("America/New_York")


class FakeData:
    def __init__(self):
        self.universe = Universe(
            sector="semis", benchmark="SMH",
            tickers=[TickerMeta(symbol="NVDA"), TickerMeta(symbol="AMD")],
        )
        self.price = 100.0

    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        if ticker not in self.universe.symbols and ticker != self.universe.benchmark:
            raise ValueError("nope")
        return pd.DataFrame({"close": [self.price]})


class FakeBroker:
    def __init__(self, positions):
        self._positions = positions

    def get_positions(self):
        return self._positions

    def get_account(self):
        return AccountSnapshot(equity=100_000.0, cash=50_000.0)


class FakeExecution:
    def __init__(self, broker):
        self.broker = broker
        self.placed = []

    def place(self, order):
        self.placed.append(order)
        return Fill(
            client_order_id=order.client_order_id, broker_order_id="x",
            ticker=order.ticker, side=order.side, status="filled",
            filled_qty=order.quantity, avg_price=order.reference_price,
            ts=datetime(2026, 6, 20, tzinfo=timezone.utc),
        )


class FakeJournal:
    def __init__(self, entries=None):
        self.entries = list(entries or [])
        self.appended = []

    def append(self, kind, ticker=None, payload=None):
        self.appended.append((kind, ticker, payload or {}))
        return None

    def read(self, ticker=None, kind=None, since=None):
        out = self.entries
        if ticker is not None:
            out = [e for e in out if e.ticker == ticker]
        if kind is not None:
            out = [e for e in out if e.kind == kind]
        return out


class FakeStrategy:
    def exit_plan(self):
        return ExitPlan(
            max_hold_days=10, stop_loss_pct=0.08, profit_target_pct=0.20,
            thesis_check_guidance="n/a",
        )


class _Settings:
    kill_switch_file = "this-file-does-not-exist"
    risk_timeframe = "1d"
    risk_lookback_days = 20


def _buy_entry(ticker, when):
    return JournalEntry(
        entry_id="1", ts=when, kind="fill", ticker=ticker, payload={"side": "buy"}
    )


def _orch(data, execution, journal, strategy):
    return Orchestrator(
        settings=_Settings(), data_layer=data, research=None, analyst=None,
        risk=None, execution=execution, journal=journal, sod_equity=None,
        strategy=strategy, clock=lambda: datetime(2026, 6, 20, 10, 0, tzinfo=ET),
        market_open=lambda now: True,
    )


def test_run_exits_places_stop_loss_sell():
    import datetime as _dt
    data = FakeData()
    data.price = 90.0  # NVDA bought at 100 -> -10% < -8% stop
    broker = FakeBroker([PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)])
    execution = FakeExecution(broker)
    journal = FakeJournal([_buy_entry("NVDA", datetime(2026, 6, 18, tzinfo=timezone.utc))])
    orch = _orch(data, execution, journal, FakeStrategy())

    fills = orch._run_exits(
        cycle_id="2026-06-20T10", as_of=None, as_of_date=_dt.date(2026, 6, 20)
    )

    assert len(fills) == 1
    order = execution.placed[0]
    assert order.side == "sell" and order.ticker == "NVDA" and order.quantity == 10
    assert order.client_order_id == "2026-06-20T10:NVDA:exit"
    assert any(k == "exit" and t == "NVDA" for k, t, _ in journal.appended)


def test_run_exits_noop_when_in_band():
    import datetime as _dt
    data = FakeData()
    data.price = 103.0  # +3%, no trigger
    broker = FakeBroker([PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)])
    execution = FakeExecution(broker)
    journal = FakeJournal([_buy_entry("NVDA", datetime(2026, 6, 18, tzinfo=timezone.utc))])
    orch = _orch(data, execution, journal, FakeStrategy())

    fills = orch._run_exits(cycle_id="c", as_of=None, as_of_date=_dt.date(2026, 6, 20))
    assert fills == [] and execution.placed == []


def test_run_exits_empty_when_no_positions():
    import datetime as _dt
    execution = FakeExecution(FakeBroker([]))
    orch = _orch(FakeData(), execution, FakeJournal(), FakeStrategy())
    assert orch._run_exits(cycle_id="c", as_of=None, as_of_date=_dt.date(2026, 6, 20)) == []


def test_run_exits_threads_as_of_into_price_marking():
    # In backtest mode the exit loop must mark point-in-time, not live.
    import datetime as _dt

    class RecordingData(FakeData):
        def __init__(self):
            super().__init__()
            self.seen_as_of = []

        def get_bars(self, ticker, timeframe, lookback, as_of=None):
            self.seen_as_of.append(as_of)
            return super().get_bars(ticker, timeframe, lookback, as_of=as_of)

    data = RecordingData()
    broker = FakeBroker([PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)])
    execution = FakeExecution(broker)
    journal = FakeJournal([_buy_entry("NVDA", datetime(2026, 6, 1, tzinfo=timezone.utc))])
    orch = _orch(data, execution, journal, FakeStrategy())

    backtest_day = _dt.date(2026, 6, 20)
    orch._run_exits(cycle_id="c", as_of=backtest_day, as_of_date=backtest_day)
    assert backtest_day in data.seen_as_of  # marked at the point-in-time date, not None


# H2 regression: exit reason propagates into the OrderRequest ----------------

def test_run_exits_order_carries_exit_reason():
    """H2: the OrderRequest placed for a stop-loss exit must carry reason='stop_loss'."""
    import datetime as _dt

    data = FakeData()
    data.price = 90.0  # NVDA bought at 100 -> -10% < -8% stop
    broker = FakeBroker([PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)])
    execution = FakeExecution(broker)
    journal = FakeJournal([_buy_entry("NVDA", datetime(2026, 6, 18, tzinfo=timezone.utc))])
    orch = _orch(data, execution, journal, FakeStrategy())

    orch._run_exits(cycle_id="2026-06-20T10", as_of=None, as_of_date=_dt.date(2026, 6, 20))

    assert len(execution.placed) == 1
    order = execution.placed[0]
    assert order.reason == "stop_loss"
