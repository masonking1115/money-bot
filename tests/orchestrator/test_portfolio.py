import pandas as pd
import pytest

from moneybot.config import TickerMeta, Universe
from moneybot.execution.models import AccountSnapshot, PositionRecord
from moneybot.orchestrator.portfolio import build_portfolio_state


class FakeData:
    """Marks NVDA at 120; raises for out-of-universe tickers like the real layer."""

    def __init__(self):
        self.universe = Universe(
            sector="semis",
            benchmark="SMH",
            tickers=[TickerMeta(symbol="NVDA"), TickerMeta(symbol="AMD")],
        )

    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        if ticker not in self.universe.symbols and ticker != self.universe.benchmark:
            raise ValueError(f"{ticker} not in universe")
        price = {"NVDA": 120.0, "SMH": 200.0}.get(ticker, 100.0)
        return pd.DataFrame({"close": [price]})


class FakeBroker:
    def __init__(self, positions, equity, cash):
        self._positions = positions
        self._equity = equity
        self._cash = cash

    def get_positions(self):
        return self._positions

    def get_account(self):
        return AccountSnapshot(equity=self._equity, cash=self._cash)


class _Settings:
    risk_timeframe = "1d"
    risk_lookback_days = 20


def test_marks_positions_to_current_price():
    broker = FakeBroker(
        positions=[PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)],
        equity=101_000.0,
        cash=99_800.0,
    )
    state = build_portfolio_state(
        broker=broker, data_layer=FakeData(), settings=_Settings(),
        as_of=None, day_pnl_pct=0.01,
    )
    assert state.equity == 101_000.0 and state.cash == 99_800.0
    assert state.day_pnl_pct == 0.01
    assert len(state.positions) == 1
    pos = state.positions[0]
    assert pos.ticker == "NVDA" and pos.shares == 10.0
    assert pos.market_value == 1_200.0  # 10 * 120 (current), not 10 * 100 (cost)


def test_short_position_marks_negative():
    broker = FakeBroker(
        positions=[PositionRecord(ticker="SMH", qty=-5.0, avg_price=210.0)],
        equity=100_000.0, cash=101_000.0,
    )
    state = build_portfolio_state(
        broker=broker, data_layer=FakeData(), settings=_Settings(),
        as_of=None, day_pnl_pct=0.0,
    )
    assert state.positions[0].market_value == -1_000.0  # -5 * 200


def test_unmarkable_ticker_falls_back_to_cost():
    # A position the data layer would reject (not in universe, not benchmark)
    broker = FakeBroker(
        positions=[PositionRecord(ticker="OLD", qty=3.0, avg_price=50.0)],
        equity=100_000.0, cash=99_850.0,
    )
    state = build_portfolio_state(
        broker=broker, data_layer=FakeData(), settings=_Settings(),
        as_of=None, day_pnl_pct=0.0,
    )
    assert state.positions[0].market_value == 150.0  # 3 * 50 (cost fallback, no crash)


def test_nonpositive_broker_equity_falls_back_to_cash_plus_marks():
    broker = FakeBroker(
        positions=[PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)],
        equity=0.0, cash=99_800.0,
    )
    state = build_portfolio_state(
        broker=broker, data_layer=FakeData(), settings=_Settings(),
        as_of=None, day_pnl_pct=0.0,
    )
    # equity must be > 0 (PortfolioState constraint): cash + marked = 99,800 + 1,200
    assert state.equity == 101_000.0


def test_nonpositive_equity_with_shorts_raises_clearly():
    # broker equity 0 AND cash + (negative) marks <= 0 -> a clear ValueError,
    # not an opaque PortfolioState ValidationError, and never a trade on a lie.
    broker = FakeBroker(
        positions=[PositionRecord(ticker="SMH", qty=-5.0, avg_price=210.0)],
        equity=0.0, cash=500.0,  # 500 + (-5 * 200) = -500
    )
    with pytest.raises(ValueError, match="non-positive"):
        build_portfolio_state(
            broker=broker, data_layer=FakeData(), settings=_Settings(),
            as_of=None, day_pnl_pct=0.0,
        )
