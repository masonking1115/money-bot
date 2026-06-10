from datetime import date

import pandas as pd

from moneybot.analyst.models import TradePlan
from moneybot.config import TickerMeta, Universe
from moneybot.risk.engine import RiskEngine
from moneybot.risk.models import PortfolioState, Position
from moneybot.strategies.catalyst_driven import CatalystDrivenLong
from moneybot.strategies.models import ExitPlan, StrategyParams


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


def test_approves_and_sizes_by_conviction(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    # flat price -> volatility 0.0 -> no vol scaling; conviction 0.5 * cap 0.10 = 0.05
    eng = _engine({"NVDA": _bars([100.0, 100.0, 100.0, 100.0])})
    out = eng.assess([_plan("NVDA", conviction=0.5)], _healthy_portfolio(),
                     as_of=date(2026, 6, 1))
    d = out.decisions[0]
    assert d.approved is True
    assert d.reference_price == 100.0
    assert d.target_weight == 0.05
    assert d.shares == 50            # 0.05 * 100_000 / 100
    assert d.target_dollars == 5_000.0
    # priced the name point-in-time via the configured timeframe/lookback
    call = eng.data.calls[0]
    assert call["ticker"] == "NVDA"
    assert call["timeframe"] == "1d"
    assert call["lookback"] == 20
    assert call["as_of"] == date(2026, 6, 1)


def test_no_pyramiding_when_already_held(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    eng = _engine({"NVDA": _bars([100.0, 100.0, 100.0])})
    port = PortfolioState(equity=100_000.0, cash=100_000.0,
                          positions=[Position(ticker="NVDA", shares=10, market_value=1_000.0)])
    out = eng.assess([_plan("NVDA")], port, as_of=date(2026, 6, 1))
    assert out.decisions[0].approved is False
    assert "already_held" in out.decisions[0].rules_fired


def test_earnings_blackout_vetoes_new_entry(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    eng = _engine({"AMD": _bars([100.0, 100.0, 100.0])})
    # AMD earnings 2026-06-11; as_of 2026-06-09 -> 2 days out, within the 3-day blackout
    out = eng.assess([_plan("AMD")], _healthy_portfolio(), as_of=date(2026, 6, 9))
    assert out.decisions[0].approved is False
    assert "earnings_blackout" in out.decisions[0].rules_fired


def test_earnings_blackout_clears_when_far_out(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    eng = _engine({"AMD": _bars([100.0, 100.0, 100.0])})
    out = eng.assess([_plan("AMD")], _healthy_portfolio(), as_of=date(2026, 6, 1))
    assert out.decisions[0].approved is True  # 10 days out, no blackout


def test_blackout_skipped_without_as_of(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    eng = _engine({"AMD": _bars([100.0, 100.0, 100.0])})
    # No as_of -> cannot measure proximity; blackout cannot fire (no fabricated clock).
    out = eng.assess([_plan("AMD")], _healthy_portfolio(), as_of=None)
    assert out.decisions[0].approved is True


def test_illiquid_name_is_vetoed(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    # 100 * 100 = 10_000 avg $-volume, far below the 5,000,000 floor
    eng = _engine({"NVDA": _bars([100.0, 100.0, 100.0], volume=100)})
    out = eng.assess([_plan("NVDA")], _healthy_portfolio(), as_of=date(2026, 6, 1))
    assert out.decisions[0].approved is False
    assert "liquidity" in out.decisions[0].rules_fired


def test_missing_price_is_vetoed_as_sanity(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    eng = _engine({})  # NVDA -> empty frame -> no price
    out = eng.assess([_plan("NVDA")], _healthy_portfolio(), as_of=date(2026, 6, 1))
    assert out.decisions[0].approved is False
    assert "sanity" in out.decisions[0].rules_fired


def test_sector_exposure_cap_downsizes(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    eng = _engine({"NVDA": _bars([100.0, 100.0, 100.0])})
    # gross already 0.58 of equity; cap 0.60 -> only 0.02 (=$2,000) headroom left
    port = PortfolioState(
        equity=100_000.0, cash=100_000.0,
        positions=[Position(ticker="AMD", shares=580, market_value=58_000.0)],
    )
    out = eng.assess([_plan("NVDA", conviction=0.5)], port, as_of=date(2026, 6, 1))
    d = out.decisions[0]
    assert d.approved is True
    assert d.shares == 20            # $2,000 / $100, downsized from the $5,000 base
    assert d.target_dollars == 2_000.0
    assert "sector_exposure_cap" in d.rules_fired


def test_sector_exposure_cap_vetoes_when_no_headroom(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    eng = _engine({"NVDA": _bars([100.0, 100.0, 100.0])})
    port = PortfolioState(
        equity=100_000.0, cash=100_000.0,
        positions=[Position(ticker="AMD", shares=600, market_value=60_000.0)],
    )
    out = eng.assess([_plan("NVDA")], port, as_of=date(2026, 6, 1))
    assert out.decisions[0].approved is False
    assert "sector_exposure_cap" in out.decisions[0].rules_fired


def test_insufficient_cash_downsizes(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    eng = _engine({"NVDA": _bars([100.0, 100.0, 100.0])})
    # cap-based size would be $5,000 but only $1,000 cash is available
    port = PortfolioState(equity=100_000.0, cash=1_000.0, positions=[])
    out = eng.assess([_plan("NVDA", conviction=0.5)], port, as_of=date(2026, 6, 1))
    d = out.decisions[0]
    assert d.approved is True
    assert d.shares == 10
    assert d.target_dollars == 1_000.0
    assert "insufficient_cash" in d.rules_fired


def test_running_exposure_consumed_across_plans(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    # cap 0.60 of $100k = $60k. Each full-conviction name wants 0.10 ($10k).
    # Start gross at 0.55 ($55k) -> only $5k headroom: NVDA takes it, AMD vetoed.
    eng = _engine({"NVDA": _bars([100.0, 100.0, 100.0]),
                   "AMD": _bars([100.0, 100.0, 100.0])})
    port = PortfolioState(
        equity=100_000.0, cash=100_000.0,
        positions=[Position(ticker="SMH", shares=550, market_value=55_000.0)],
    )
    out = eng.assess([_plan("NVDA", conviction=1.0), _plan("AMD", conviction=1.0)],
                     port, as_of=date(2026, 6, 1))
    nvda, amd = out.decisions
    assert nvda.approved is True and nvda.target_dollars == 5_000.0
    assert amd.approved is False and "sector_exposure_cap" in amd.rules_fired


def test_out_of_universe_plan_is_vetoed_not_crashing(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    # TSLA is not in the universe (NVDA, AMD). The real DataLayer.get_bars would
    # raise ValueError for it; the engine must veto, never crash, and never price it.
    eng = _engine({"NVDA": _bars([100.0, 100.0, 100.0])})
    out = eng.assess([_plan("TSLA")], _healthy_portfolio(), as_of=date(2026, 6, 1))
    assert out.decisions[0].approved is False
    assert "sanity" in out.decisions[0].rules_fired
    assert all(c["ticker"] != "TSLA" for c in eng.data.calls)


def test_volatility_scaling_reduces_size_end_to_end(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    # closes 100,110,99 -> realized_volatility = sqrt(0.02) ~ 0.1414, far above the
    # 0.02 target, so a full-conviction name is scaled DOWN from the 0.10 cap.
    eng = _engine({"NVDA": _bars([100.0, 110.0, 99.0])})
    out = eng.assess([_plan("NVDA", conviction=1.0)], _healthy_portfolio(),
                     as_of=date(2026, 6, 1))
    d = out.decisions[0]
    assert d.approved is True
    assert d.reference_price == 99.0
    assert d.target_weight < 0.10          # vol-scaling trimmed it below the cap
    assert d.shares == 14                   # 0.10 * sqrt(0.02) * 100_000 = $1414.2 -> //99 = 14
    assert d.rules_fired == []              # vol-scaling is intrinsic to sizing, not a cap rule


def test_no_hedge_when_disabled(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    eng = _engine({"NVDA": _bars([100.0, 100.0, 100.0])})  # default params: hedge off
    out = eng.assess([_plan("NVDA")], _healthy_portfolio(), as_of=date(2026, 6, 1))
    assert out.hedge is None


def test_hedge_offsets_gross_long_when_enabled(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    strategy = CatalystDrivenLong(StrategyParams(hedge_enabled=True))
    eng = _engine(
        {"NVDA": _bars([100.0, 100.0, 100.0]), "SMH": _bars([50.0, 50.0, 50.0])},
        strategy=strategy,
    )
    # one approved name: 0.5 conviction * 0.10 cap = $5,000 long, no prior positions
    out = eng.assess([_plan("NVDA", conviction=0.5)], _healthy_portfolio(),
                     as_of=date(2026, 6, 1))
    assert out.hedge is not None
    assert out.hedge.ticker == "SMH"
    assert out.hedge.side == "short"
    # gross long $5,000 * hedge_ratio 0.5 = $2,500 hedged / $50 SMH = 50 shares
    assert out.hedge.shares == 50
    assert out.hedge.dollars == 2_500.0


def test_no_hedge_when_no_long_exposure(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    strategy = CatalystDrivenLong(StrategyParams(hedge_enabled=True))
    eng = _engine({"NVDA": _bars([100.0, 100.0, 100.0], volume=100),  # illiquid -> vetoed
                   "SMH": _bars([50.0, 50.0, 50.0])}, strategy=strategy)
    out = eng.assess([_plan("NVDA")], _healthy_portfolio(), as_of=date(2026, 6, 1))
    assert out.hedge is None  # nothing long -> nothing to hedge


def test_risk_engine_is_exported_from_package():
    from moneybot.risk import RiskEngine as Exported
    assert Exported is RiskEngine


def test_hedge_includes_existing_long_and_excludes_shorts(monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    strategy = CatalystDrivenLong(StrategyParams(hedge_enabled=True))
    eng = _engine(
        {"NVDA": _bars([100.0, 100.0, 100.0]), "SMH": _bars([50.0, 50.0, 50.0])},
        strategy=strategy,
    )
    port = PortfolioState(
        equity=100_000.0, cash=100_000.0,
        positions=[
            Position(ticker="AMD", shares=300, market_value=30_000.0),       # long, counts
            Position(ticker="SHORTX", shares=-100, market_value=-5_000.0),   # short, excluded
        ],
    )
    out = eng.assess([_plan("NVDA", conviction=0.5)], port, as_of=date(2026, 6, 1))
    # gross long = existing $30,000 (the short is excluded) + new $5,000 = $35,000
    # hedge = $35,000 * 0.5 hedge_ratio = $17,500 / $50 SMH = 350 shares
    assert out.hedge is not None
    assert out.hedge.shares == 350
    assert out.hedge.dollars == 17_500.0
    # the benchmark was priced point-in-time (as_of threaded through)
    smh_call = next(c for c in eng.data.calls if c["ticker"] == "SMH")
    assert smh_call["as_of"] == date(2026, 6, 1)
