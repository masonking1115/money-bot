from datetime import date

import pandas as pd

from moneybot.analyst.models import TradePlan
from moneybot.config import TickerMeta, Universe
from moneybot.risk.engine import RiskEngine
from moneybot.risk.models import PortfolioState, Position  # noqa: F401 – used in Tasks 7-8
from moneybot.strategies.catalyst_driven import CatalystDrivenLong
from moneybot.strategies.models import ExitPlan, StrategyParams  # noqa: F401 – used in Tasks 7-8


def _exit():
    return ExitPlan(max_hold_days=10, stop_loss_pct=0.08, profit_target_pct=0.20,
                    thesis_check_guidance="re-read filings")


def _plan(ticker, conviction=0.5):
    return TradePlan(ticker=ticker, action="buy", conviction=conviction,
                     thesis="catalyst", score=0.5, signal_ref="sig-1",
                     exit_plan=_exit(), analyst_note="ok")


def _universe():
    return Universe(sector="semiconductors", benchmark="SMH",
                    tickers=[TickerMeta(symbol="NVDA"),
                             TickerMeta(symbol="AMD", earnings_date=date(2026, 6, 11))])


def _bars(prices, volume=10_000_000):
    n = len(prices)
    return pd.DataFrame({
        "ts": pd.date_range("2026-05-01", periods=n, freq="D"),
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": [volume] * n,
    })


class FakeData:
    """Duck-typed DataLayer: canned bars per symbol + a Universe."""

    def __init__(self, bars_by_symbol, universe):
        self._bars = bars_by_symbol
        self.universe = universe
        self.calls = []

    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        self.calls.append({"ticker": ticker, "timeframe": timeframe,
                           "lookback": lookback, "as_of": as_of})
        return self._bars.get(ticker, pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "volume"]))


def _settings():
    from moneybot.config import Settings
    return Settings(kill_switch_file="this-file-does-not-exist")


def _engine(bars_by_symbol, *, strategy=None, settings=None):
    data = FakeData(bars_by_symbol, _universe())
    return RiskEngine(data_layer=data, strategy=strategy or CatalystDrivenLong(),
                      settings=settings or _settings())


def _healthy_portfolio():
    return PortfolioState(equity=100_000.0, cash=100_000.0, positions=[])


def test_kill_switch_vetoes_everything_and_halts(monkeypatch):
    monkeypatch.setenv("MONEYBOT_KILL_SWITCH", "1")
    eng = _engine({"NVDA": _bars([100.0, 101.0, 102.0])})
    out = eng.assess([_plan("NVDA")], _healthy_portfolio(), as_of=date(2026, 6, 1))
    assert out.halted is True
    assert len(out.decisions) == 1
    assert out.decisions[0].approved is False
    assert "kill_switch" in out.decisions[0].rules_fired
    # No price reads happen once the kill switch is engaged.
    assert eng.data.calls == []


def test_daily_loss_circuit_breaker_halts_new_entries(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    eng = _engine({"NVDA": _bars([100.0, 101.0, 102.0])})
    port = PortfolioState(equity=100_000.0, cash=100_000.0, positions=[], day_pnl_pct=-0.05)
    out = eng.assess([_plan("NVDA")], port, as_of=date(2026, 6, 1))
    assert out.halted is True
    assert out.decisions[0].approved is False
    assert "daily_loss_circuit_breaker" in out.decisions[0].rules_fired


def test_circuit_breaker_not_tripped_above_floor(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    eng = _engine({"NVDA": _bars([100.0, 100.0, 100.0])})
    port = PortfolioState(equity=100_000.0, cash=100_000.0, positions=[], day_pnl_pct=-0.02)
    out = eng.assess([_plan("NVDA")], port, as_of=date(2026, 6, 1))
    assert out.halted is False  # -2% is above the -3% floor
