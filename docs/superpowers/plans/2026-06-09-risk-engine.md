# Risk Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic Risk Engine — pure Python, no LLM — that turns the Analyst's `TradePlan`s into sized, approved/downsized/vetoed orders against hard-coded limits the agents cannot bypass.

**Architecture:** A thin `RiskEngine` coordinator runs a fixed rule pipeline (kill switch → daily-loss circuit breaker → per-plan: no-pyramiding → earnings blackout → price/liquidity sanity → conviction × volatility sizing → sector-exposure & cash caps), then optionally computes a benchmark hedge. All sizing/volatility math lives in pure, series-in/scalar-out modules (mirroring `analyst/relative_strength.py`); all price access goes through `DataLayer.get_bars` with point-in-time `as_of`. Per-name and sector caps come from the active strategy's `StrategyParams`; operational limits (daily-loss floor, blackout days, min liquidity, vol target, hedge ratio, kill-switch file) come from `Settings`. Every decision records the rule(s) that fired.

**Tech Stack:** Python 3, Pydantic v2, pandas (bars only), uv toolchain (`uv run pytest`, `uv run ruff check`), ruff line-length 100. Production modules use `from __future__ import annotations`; test files omit it. No test touches the network or a real clock.

---

## File Structure

- Create: `src/moneybot/risk/__init__.py` — package marker; re-exports `RiskEngine`.
- Create: `src/moneybot/risk/models.py` — `Position`, `PortfolioState`, `RiskDecision`, `HedgeOrder`, `RiskAssessment`.
- Create: `src/moneybot/risk/metrics.py` — PURE `realized_volatility`, `average_dollar_volume`.
- Create: `src/moneybot/risk/sizing.py` — PURE `target_weight`.
- Create: `src/moneybot/risk/kill_switch.py` — `kill_switch_active`.
- Create: `src/moneybot/risk/engine.py` — `RiskEngine` coordinator + the rule pipeline.
- Create: `src/moneybot/risk/factory.py` — `build_risk_engine`.
- Modify: `src/moneybot/config.py` — add Risk Engine settings.
- Modify: `README.md` — add a plain-prose "Phase 7: risk engine" bullet.
- Test: `tests/risk/__init__.py`, `tests/risk/test_models.py`, `tests/risk/test_metrics.py`, `tests/risk/test_sizing.py`, `tests/risk/test_kill_switch.py`, `tests/risk/test_config_risk.py`, `tests/risk/test_engine.py`, `tests/risk/test_factory.py`.

### Interfaces this plan consumes (already in the codebase — do not redefine)

- `moneybot.analyst.models.TradePlan` — fields used here: `ticker: str`, `action: Literal["buy"]`, `conviction: float`, `thesis: str`, `score: float`, `signal_ref: str | None`, `exit_plan: ExitPlan`, `analyst_note: str`, `risk_flags: list[str]`.
- `moneybot.strategies.models.StrategyParams` — `max_position_pct` (default `0.10`), `max_sector_exposure_pct` (default `0.60`), `hedge_enabled` (default `False`).
- `moneybot.strategies.base.Strategy` — `parameters() -> StrategyParams`.
- `moneybot.config.Settings`, `moneybot.config.Universe`, `moneybot.config.TickerMeta` (`symbol`, `earnings_date: date | None`).
- `moneybot.data_layer.DataLayer` — `get_bars(ticker, timeframe, lookback, as_of=None) -> pd.DataFrame` (columns `ts, open, high, low, close, volume`, oldest-first); `.universe` (`.symbols`, `.benchmark`, `.sector`, `.get(symbol)`).
- `moneybot.strategies.registry` — `registry.get(name) -> Strategy` (import `moneybot.strategies` for the registration side-effect).

---

## Task 1: Risk Engine settings

**Files:**
- Modify: `src/moneybot/config.py` (after the `# Analyst` block, lines 37–40)
- Test: `tests/risk/__init__.py`, `tests/risk/test_config_risk.py`

- [ ] **Step 1: Create the test package marker**

Create `tests/risk/__init__.py` as an empty file (the repo uses package dirs for tests).

```python
```

- [ ] **Step 2: Write the failing test**

Create `tests/risk/test_config_risk.py`:

```python
from moneybot.config import Settings


def test_risk_settings_have_sane_defaults():
    s = Settings()
    assert s.daily_loss_limit_pct == 0.03
    assert s.earnings_blackout_days == 3
    assert s.min_dollar_volume == 5_000_000.0
    assert s.target_volatility == 0.02
    assert s.hedge_ratio == 0.5
    assert s.risk_timeframe == "1d"
    assert s.risk_lookback_days == 20
    assert s.kill_switch_file == "KILL_SWITCH"


def test_risk_settings_override_from_env(monkeypatch):
    monkeypatch.setenv("MONEYBOT_DAILY_LOSS_LIMIT_PCT", "0.05")
    monkeypatch.setenv("MONEYBOT_RISK_LOOKBACK_DAYS", "30")
    s = Settings()
    assert s.daily_loss_limit_pct == 0.05
    assert s.risk_lookback_days == 30
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/risk/test_config_risk.py -v`
Expected: FAIL — `AttributeError`/assertion on the missing `daily_loss_limit_pct` field.

- [ ] **Step 4: Add the settings**

In `src/moneybot/config.py`, immediately after the Analyst block (the line `rs_timeframe: str = "1d"      # ...`), add:

```python

    # Risk Engine
    daily_loss_limit_pct: float = 0.03   # halt new entries when day P&L <= -3% of equity
    earnings_blackout_days: int = 3      # no new entry within N days before a known earnings date
    min_dollar_volume: float = 5_000_000.0  # min avg daily $-volume for a name to be tradeable
    target_volatility: float = 0.02      # per-bar return-stddev target for volatility-scaling
    hedge_ratio: float = 0.5             # fraction of gross long hedged via the benchmark (if enabled)
    risk_timeframe: str = "1d"           # bar timeframe for risk metrics (vol/liquidity/price)
    risk_lookback_days: int = 20         # bar lookback for risk metrics
    kill_switch_file: str = "KILL_SWITCH"  # if this file exists, all trading halts immediately
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/risk/test_config_risk.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Lint**

Run: `uv run ruff check src/moneybot/config.py tests/risk/test_config_risk.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/moneybot/config.py tests/risk/__init__.py tests/risk/test_config_risk.py
git commit -m "feat(risk): add Risk Engine settings"
```

---

## Task 2: Risk Engine models

**Files:**
- Create: `src/moneybot/risk/__init__.py`, `src/moneybot/risk/models.py`
- Test: `tests/risk/test_models.py`

- [ ] **Step 1: Create the package marker**

Create `src/moneybot/risk/__init__.py`. Leave it empty for now (Task 8 adds the `RiskEngine` re-export — do not import it yet, the module does not exist).

```python
"""Risk Engine: deterministic sizing and hard-limit enforcement (no LLM)."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/risk/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from moneybot.risk.models import (
    HedgeOrder,
    PortfolioState,
    Position,
    RiskAssessment,
    RiskDecision,
)


def test_portfolio_exposure_properties():
    p = PortfolioState(
        equity=100_000.0,
        cash=40_000.0,
        positions=[
            Position(ticker="NVDA", shares=100, market_value=40_000.0),
            Position(ticker="AMD", shares=50, market_value=20_000.0),
        ],
    )
    assert p.long_market_value == 60_000.0
    assert p.gross_exposure_pct == pytest.approx(0.60)


def test_portfolio_exposure_ignores_non_positive_market_values():
    p = PortfolioState(
        equity=100_000.0,
        cash=100_000.0,
        positions=[Position(ticker="NVDA", shares=0, market_value=0.0)],
    )
    assert p.long_market_value == 0.0
    assert p.gross_exposure_pct == 0.0


def test_portfolio_requires_positive_equity():
    with pytest.raises(ValidationError):
        PortfolioState(equity=0.0, cash=0.0)


def test_assessment_approved_filters_decisions():
    approved = RiskDecision(ticker="NVDA", approved=True, target_weight=0.05,
                            target_dollars=5_000.0, shares=50, reference_price=100.0,
                            reasoning="approved")
    vetoed = RiskDecision(ticker="AMD", approved=False, rules_fired=["liquidity"],
                          reasoning="too illiquid")
    a = RiskAssessment(decisions=[approved, vetoed])
    assert a.approved == [approved]
    assert a.halted is False
    assert a.hedge is None


def test_hedge_order_is_short_only():
    h = HedgeOrder(ticker="SMH", side="short", shares=50, dollars=2_500.0)
    assert h.side == "short"
    with pytest.raises(ValidationError):
        HedgeOrder(ticker="SMH", side="long", shares=1, dollars=1.0)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/risk/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.risk.models'`.

- [ ] **Step 4: Write the models**

Create `src/moneybot/risk/models.py`:

```python
"""Input and output types for the Risk Engine.

PortfolioState / Position are the account snapshot the orchestrator (Plan 8)
supplies. RiskDecision is the verdict for one TradePlan (approved, downsized, or
vetoed) and always records the rule(s) that fired. RiskAssessment bundles a
cycle's decisions with the halt flag and an optional benchmark hedge.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Position(BaseModel):
    """One open position in the account snapshot."""

    ticker: str
    shares: float
    market_value: float  # current market value (shares * current price)


class PortfolioState(BaseModel):
    """Account snapshot the Risk Engine sizes against."""

    equity: float = Field(gt=0)  # total account value (cash + positions)
    cash: float
    positions: list[Position] = Field(default_factory=list)
    day_pnl_pct: float = 0.0  # today's P&L as a fraction of starting equity (negative = loss)

    @property
    def long_market_value(self) -> float:
        return sum(p.market_value for p in self.positions if p.market_value > 0)

    @property
    def gross_exposure_pct(self) -> float:
        return self.long_market_value / self.equity if self.equity else 0.0


class RiskDecision(BaseModel):
    """The Risk Engine's verdict on a single TradePlan."""

    ticker: str
    approved: bool
    target_weight: float = 0.0  # approved fraction of equity (0 if vetoed)
    target_dollars: float = 0.0  # shares * reference_price actually deployed
    shares: int = 0
    reference_price: float | None = None
    rules_fired: list[str] = Field(default_factory=list)  # rules that downsized or vetoed
    reasoning: str


class HedgeOrder(BaseModel):
    """An offsetting benchmark position to neutralize sector beta (when enabled)."""

    ticker: str
    side: Literal["short"]
    shares: int
    dollars: float


class RiskAssessment(BaseModel):
    """The full result of assessing one cycle's trade plans."""

    decisions: list[RiskDecision]
    halted: bool = False  # true when a global gate (kill switch / circuit breaker) stopped entries
    hedge: HedgeOrder | None = None

    @property
    def approved(self) -> list[RiskDecision]:
        return [d for d in self.decisions if d.approved]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/risk/test_models.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Lint**

Run: `uv run ruff check src/moneybot/risk/ tests/risk/test_models.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/moneybot/risk/__init__.py src/moneybot/risk/models.py tests/risk/test_models.py
git commit -m "feat(risk): add Risk Engine I/O models"
```

---

## Task 3: Pure risk metrics (volatility + liquidity)

**Files:**
- Create: `src/moneybot/risk/metrics.py`
- Test: `tests/risk/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/risk/test_metrics.py`:

```python
import pytest

from moneybot.risk.metrics import average_dollar_volume, realized_volatility


def test_realized_volatility_of_constant_series_is_zero():
    assert realized_volatility([100.0, 100.0, 100.0, 100.0]) == 0.0


def test_realized_volatility_is_sample_stddev_of_returns():
    # returns: +0.10 then -0.10/1.10 ... use a hand-checkable series
    # closes 100, 110, 99 -> returns 0.10 and -0.10 -> mean 0.0, sample var = (0.01+0.01)/1 = 0.02
    vol = realized_volatility([100.0, 110.0, 99.0])
    assert vol == pytest.approx(0.02**0.5)


def test_realized_volatility_needs_at_least_three_closes():
    assert realized_volatility([100.0, 110.0]) is None
    assert realized_volatility([]) is None


def test_realized_volatility_skips_none_values():
    assert realized_volatility([100.0, None, 110.0, 99.0]) == pytest.approx(0.02**0.5)


def test_average_dollar_volume_means_price_times_volume():
    # (100*10 + 200*5) / 2 = (1000 + 1000)/2 = 1000
    assert average_dollar_volume([100.0, 200.0], [10, 5]) == 1000.0


def test_average_dollar_volume_none_when_no_pairs():
    assert average_dollar_volume([], []) is None
    assert average_dollar_volume([None], [None]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/risk/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.risk.metrics'`.

- [ ] **Step 3: Write the metrics**

Create `src/moneybot/risk/metrics.py`:

```python
"""Pure price/volume math for the Risk Engine.

No I/O and no pandas — callers pass plain sequences so these are trivially
unit-testable (mirrors analyst/relative_strength.py). Returns None when a metric
is not computable so the engine can veto on missing data rather than guess.
"""

from __future__ import annotations

from collections.abc import Sequence


def realized_volatility(closes: Sequence[float | None]) -> float | None:
    """Sample standard deviation of period-over-period simple returns.

    None if fewer than three valid closes (need >=2 returns for a sample stddev).
    A flat series returns 0.0.
    """
    vals = [c for c in closes if c is not None]
    if len(vals) < 3:
        return None
    returns = [(vals[i] / vals[i - 1]) - 1.0 for i in range(1, len(vals)) if vals[i - 1] != 0]
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return var**0.5


def average_dollar_volume(
    closes: Sequence[float | None],
    volumes: Sequence[float | None],
) -> float | None:
    """Mean of close*volume over bars where both are present. None if no such bar."""
    pairs = [(c, v) for c, v in zip(closes, volumes) if c is not None and v is not None]
    if not pairs:
        return None
    return sum(c * v for c, v in pairs) / len(pairs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/risk/test_metrics.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/moneybot/risk/metrics.py tests/risk/test_metrics.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/risk/metrics.py tests/risk/test_metrics.py
git commit -m "feat(risk): add pure volatility + liquidity metrics"
```

---

## Task 4: Pure position sizing

**Files:**
- Create: `src/moneybot/risk/sizing.py`
- Test: `tests/risk/test_sizing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/risk/test_sizing.py`:

```python
import pytest

from moneybot.risk.sizing import target_weight


def test_base_size_is_conviction_times_cap_when_vol_unknown():
    # 0.5 conviction * 0.10 cap = 0.05, no volatility info -> no scaling
    w = target_weight(conviction=0.5, volatility=None,
                      max_position_pct=0.10, target_volatility=0.02)
    assert w == pytest.approx(0.05)


def test_full_conviction_hits_the_cap():
    w = target_weight(conviction=1.0, volatility=None,
                      max_position_pct=0.10, target_volatility=0.02)
    assert w == pytest.approx(0.10)


def test_high_volatility_scales_size_down():
    # vol 0.04 vs target 0.02 -> scale 0.5 -> 0.10 * 0.5 = 0.05
    w = target_weight(conviction=1.0, volatility=0.04,
                      max_position_pct=0.10, target_volatility=0.02)
    assert w == pytest.approx(0.05)


def test_low_volatility_is_not_scaled_up_past_base():
    # calmer than target -> scale clamped to 1.0, base unchanged
    w = target_weight(conviction=1.0, volatility=0.01,
                      max_position_pct=0.10, target_volatility=0.02)
    assert w == pytest.approx(0.10)


def test_zero_or_negative_inputs_floor_at_zero():
    assert target_weight(conviction=0.0, volatility=0.02,
                         max_position_pct=0.10, target_volatility=0.02) == 0.0
    # zero volatility is treated as "no usable scaling" -> base size
    assert target_weight(conviction=0.5, volatility=0.0,
                         max_position_pct=0.10, target_volatility=0.02) == pytest.approx(0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/risk/test_sizing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.risk.sizing'`.

- [ ] **Step 3: Write the sizing function**

Create `src/moneybot/risk/sizing.py`:

```python
"""Pure position-sizing math for the Risk Engine.

Conviction sets the base slice (capped at max_position_pct); volatility-scaling
then trims names more volatile than the target so no single name dominates
portfolio risk. Scaling never increases a position past its conviction-capped
base (we trim risk, we never lever up a calm name). Output is a fraction of
equity in [0, max_position_pct].
"""

from __future__ import annotations


def target_weight(
    *,
    conviction: float,
    volatility: float | None,
    max_position_pct: float,
    target_volatility: float,
) -> float:
    """Fraction of equity to allocate to one name."""
    base = conviction * max_position_pct
    if volatility is not None and volatility > 0:
        scale = min(1.0, target_volatility / volatility)
        base *= scale
    return max(0.0, min(base, max_position_pct))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/risk/test_sizing.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/moneybot/risk/sizing.py tests/risk/test_sizing.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/risk/sizing.py tests/risk/test_sizing.py
git commit -m "feat(risk): add pure conviction + volatility position sizing"
```

---

## Task 5: Kill switch

**Files:**
- Create: `src/moneybot/risk/kill_switch.py`
- Test: `tests/risk/test_kill_switch.py`

- [ ] **Step 1: Write the failing test**

Create `tests/risk/test_kill_switch.py`:

```python
from moneybot.config import Settings
from moneybot.risk.kill_switch import kill_switch_active


def test_inactive_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    s = Settings(kill_switch_file=str(tmp_path / "nope"))
    assert kill_switch_active(s) is False


def test_active_when_env_flag_truthy(tmp_path, monkeypatch):
    monkeypatch.setenv("MONEYBOT_KILL_SWITCH", "1")
    s = Settings(kill_switch_file=str(tmp_path / "nope"))
    assert kill_switch_active(s) is True


def test_env_flag_false_string_stays_inactive(tmp_path, monkeypatch):
    monkeypatch.setenv("MONEYBOT_KILL_SWITCH", "false")
    s = Settings(kill_switch_file=str(tmp_path / "nope"))
    assert kill_switch_active(s) is False


def test_active_when_file_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("MONEYBOT_KILL_SWITCH", raising=False)
    flag = tmp_path / "KILL_SWITCH"
    flag.write_text("halt")
    s = Settings(kill_switch_file=str(flag))
    assert kill_switch_active(s) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/risk/test_kill_switch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.risk.kill_switch'`.

- [ ] **Step 3: Write the kill switch**

Create `src/moneybot/risk/kill_switch.py`:

```python
"""The kill switch: a single, always-checkable flag that halts all trading.

Active when either the MONEYBOT_KILL_SWITCH env var is truthy or the configured
file exists on disk. Kept independent of the rest of the engine so it can be
tripped by an operator out-of-band (touch a file) without code or config change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moneybot.config import Settings

_TRUTHY = {"1", "true", "yes", "on"}


def kill_switch_active(settings: Settings) -> bool:
    if os.environ.get("MONEYBOT_KILL_SWITCH", "").strip().lower() in _TRUTHY:
        return True
    return Path(settings.kill_switch_file).exists()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/risk/test_kill_switch.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/moneybot/risk/kill_switch.py tests/risk/test_kill_switch.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/risk/kill_switch.py tests/risk/test_kill_switch.py
git commit -m "feat(risk): add kill switch (env flag or file)"
```

---

## Task 6: RiskEngine — global gates (kill switch + daily-loss circuit breaker)

This task creates `engine.py` with the `RiskEngine` class and the two **global** gates that veto *every* plan and set `halted=True`. Per-plan rules and the hedge come in Tasks 7–8. The test file defines reusable fakes (`_universe`, `FakeData`, `_bars`) used again in Tasks 7–8.

**Files:**
- Create: `src/moneybot/risk/engine.py`
- Test: `tests/risk/test_engine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/risk/test_engine.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/risk/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.risk.engine'`.

- [ ] **Step 3: Write the engine skeleton + global gates**

Create `src/moneybot/risk/engine.py`:

```python
"""RiskEngine: the deterministic layer the agents cannot bypass.

Pure Python, no LLM. Given the Analyst's TradePlans and a portfolio snapshot, it
runs a fixed rule pipeline and emits a RiskAssessment. Two GLOBAL gates can stop
the whole cycle (kill switch, daily-loss circuit breaker); the remaining rules
are per-plan (Task 7) and an optional hedge is computed last (Task 8). Per-name
and sector caps come from the active strategy's parameters; operational limits
come from Settings. Every decision records the rule(s) that fired.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from moneybot.risk.kill_switch import kill_switch_active
from moneybot.risk.models import RiskAssessment, RiskDecision

if TYPE_CHECKING:
    from moneybot.analyst.models import TradePlan
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.risk.models import PortfolioState
    from moneybot.strategies.base import Strategy


class RiskEngine:
    def __init__(
        self,
        *,
        data_layer: DataLayer,
        strategy: Strategy,
        settings: Settings,
    ) -> None:
        self.data = data_layer
        self.strategy = strategy
        self.settings = settings

    @staticmethod
    def _veto(plan: TradePlan, rule: str, reasoning: str) -> RiskDecision:
        return RiskDecision(
            ticker=plan.ticker, approved=False, rules_fired=[rule], reasoning=reasoning
        )

    def assess(
        self,
        plans: list[TradePlan],
        portfolio: PortfolioState,
        as_of: date | None = None,
    ) -> RiskAssessment:
        """Run the rule pipeline over the cycle's plans and return verdicts."""
        if kill_switch_active(self.settings):
            return RiskAssessment(
                decisions=[
                    self._veto(p, "kill_switch", "kill switch engaged") for p in plans
                ],
                halted=True,
            )

        if portfolio.day_pnl_pct <= -self.settings.daily_loss_limit_pct:
            return RiskAssessment(
                decisions=[
                    self._veto(
                        p,
                        "daily_loss_circuit_breaker",
                        f"day P&L {portfolio.day_pnl_pct:.2%} at/under the "
                        f"-{self.settings.daily_loss_limit_pct:.0%} floor",
                    )
                    for p in plans
                ],
                halted=True,
            )

        # Per-plan pipeline and hedge are added in Tasks 7-8.
        return RiskAssessment(decisions=[], halted=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/risk/test_engine.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/moneybot/risk/engine.py tests/risk/test_engine.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/risk/engine.py tests/risk/test_engine.py
git commit -m "feat(risk): RiskEngine global gates (kill switch + circuit breaker)"
```

---

## Task 7: RiskEngine — per-plan rule pipeline

Adds the per-plan rules to `assess`: no-pyramiding, earnings blackout, price/liquidity sanity, conviction × volatility sizing, and the running sector-exposure + cash caps (which downsize). Plans are processed in arrival order (the Analyst already ranks them); each approval consumes exposure headroom for the next.

**Files:**
- Modify: `src/moneybot/risk/engine.py`
- Test: `tests/risk/test_engine.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/risk/test_engine.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/risk/test_engine.py -v`
Expected: FAIL — the new tests fail because `assess` currently returns `decisions=[]` for the non-halt path (e.g. `IndexError` / empty decisions).

- [ ] **Step 3: Implement the per-plan pipeline**

In `src/moneybot/risk/engine.py`, add the new imports at the top of the existing import block — replace:

```python
from moneybot.risk.kill_switch import kill_switch_active
from moneybot.risk.models import RiskAssessment, RiskDecision
```

with:

```python
from moneybot.risk.kill_switch import kill_switch_active
from moneybot.risk.metrics import average_dollar_volume, realized_volatility
from moneybot.risk.models import RiskAssessment, RiskDecision
from moneybot.risk.sizing import target_weight
```

Then replace the placeholder tail of `assess` — the lines:

```python
        # Per-plan pipeline and hedge are added in Tasks 7-8.
        return RiskAssessment(decisions=[], halted=False)
```

with:

```python
        params = self.strategy.parameters()
        held = {pos.ticker for pos in portfolio.positions}
        running_gross = portfolio.gross_exposure_pct

        decisions: list[RiskDecision] = []
        for plan in plans:
            decision = self._assess_plan(
                plan, portfolio, params, held, running_gross, as_of
            )
            decisions.append(decision)
            if decision.approved:
                running_gross += decision.target_weight
                held.add(plan.ticker)

        return RiskAssessment(decisions=decisions, halted=False)
```

Then add these methods to the `RiskEngine` class (after `assess`):

```python
    def _in_earnings_blackout(self, ticker: str, as_of: date | None) -> bool:
        """True when a known earnings date is today..N days ahead of as_of.

        Without an as_of we cannot measure proximity, and we never fabricate a
        clock — so the blackout simply cannot fire in that case.
        """
        if as_of is None:
            return False
        try:
            meta = self.data.universe.get(ticker)
        except KeyError:
            return False
        earnings = meta.earnings_date
        if earnings is None:
            return False
        days = (earnings - as_of).days
        return 0 <= days <= self.settings.earnings_blackout_days

    def _assess_plan(
        self,
        plan: TradePlan,
        portfolio: PortfolioState,
        params,  # StrategyParams
        held: set[str],
        running_gross: float,
        as_of: date | None,
    ) -> RiskDecision:
        if plan.ticker in held:
            return self._veto(plan, "already_held", "position already open; no pyramiding")

        if self._in_earnings_blackout(plan.ticker, as_of):
            return self._veto(
                plan, "earnings_blackout", "within the earnings blackout window"
            )

        bars = self.data.get_bars(
            plan.ticker,
            self.settings.risk_timeframe,
            self.settings.risk_lookback_days,
            as_of=as_of,
        )
        closes = [] if bars.empty else bars["close"].tolist()
        volumes = [] if bars.empty else bars["volume"].tolist()
        price = closes[-1] if closes else None
        if price is None or price <= 0:
            return self._veto(plan, "sanity", "no valid reference price")

        adv = average_dollar_volume(closes, volumes)
        if adv is None or adv < self.settings.min_dollar_volume:
            return self._veto(plan, "liquidity", "below the minimum $-volume floor")

        weight = target_weight(
            conviction=plan.conviction,
            volatility=realized_volatility(closes),
            max_position_pct=params.max_position_pct,
            target_volatility=self.settings.target_volatility,
        )
        if weight <= 0:
            return self._veto(plan, "sizing", "computed a zero position size")

        rules: list[str] = []
        target_dollars = weight * portfolio.equity

        sector_headroom = (params.max_sector_exposure_pct - running_gross) * portfolio.equity
        if sector_headroom <= 0:
            return self._veto(plan, "sector_exposure_cap", "no sector exposure headroom")
        if sector_headroom < target_dollars:
            target_dollars = sector_headroom
            rules.append("sector_exposure_cap")

        if portfolio.cash < target_dollars:
            target_dollars = portfolio.cash
            rules.append("insufficient_cash")

        shares = int(target_dollars // price)
        if shares <= 0:
            return self._veto(
                plan, rules[-1] if rules else "sanity", "rounds to zero shares"
            )

        actual_dollars = shares * price
        return RiskDecision(
            ticker=plan.ticker,
            approved=True,
            target_weight=actual_dollars / portfolio.equity,
            target_dollars=actual_dollars,
            shares=shares,
            reference_price=price,
            rules_fired=rules,
            reasoning="approved within limits",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/risk/test_engine.py -v`
Expected: PASS (all engine tests pass, including the Task 6 gates).

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/moneybot/risk/engine.py tests/risk/test_engine.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/risk/engine.py tests/risk/test_engine.py
git commit -m "feat(risk): per-plan rule pipeline (pyramiding, blackout, liquidity, sizing, caps)"
```

---

## Task 8: RiskEngine — optional benchmark hedge + package export

Adds the optional SMH hedge (computed only when the strategy's `hedge_enabled` is true) and wires the `RiskEngine` re-export.

**Files:**
- Modify: `src/moneybot/risk/engine.py`
- Modify: `src/moneybot/risk/__init__.py`
- Test: `tests/risk/test_engine.py` (append), `tests/risk/test_models.py` is unchanged

- [ ] **Step 1: Append the failing tests**

Append to `tests/risk/test_engine.py`:

```python
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
```

Also append a test that confirms the package export:

```python
def test_risk_engine_is_exported_from_package():
    from moneybot.risk import RiskEngine as Exported
    assert Exported is RiskEngine
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/risk/test_engine.py -v`
Expected: FAIL — `out.hedge` is `None` even when enabled (hedge not computed yet); `from moneybot.risk import RiskEngine` raises `ImportError`.

- [ ] **Step 3: Implement the hedge**

In `src/moneybot/risk/engine.py`, add `HedgeOrder` to the models import — replace:

```python
from moneybot.risk.models import RiskAssessment, RiskDecision
```

with:

```python
from moneybot.risk.models import HedgeOrder, RiskAssessment, RiskDecision
```

Then, in `assess`, replace the final return of the non-halt path:

```python
        return RiskAssessment(decisions=decisions, halted=False)
```

with:

```python
        hedge = None
        if params.hedge_enabled:
            hedge = self._hedge(portfolio, decisions, as_of)
        return RiskAssessment(decisions=decisions, halted=False, hedge=hedge)
```

Then add the `_hedge` method to the `RiskEngine` class (after `_assess_plan`):

```python
    def _hedge(
        self,
        portfolio: PortfolioState,
        decisions: list[RiskDecision],
        as_of: date | None,
    ) -> HedgeOrder | None:
        """Short the benchmark to offset a fraction of gross long exposure.

        Gross long = existing long market value + newly approved dollars. Returns
        None when there is nothing to hedge or the benchmark cannot be priced.
        """
        new_long = sum(d.target_dollars for d in decisions if d.approved)
        gross_long = portfolio.long_market_value + new_long
        if gross_long <= 0:
            return None

        benchmark = self.data.universe.benchmark
        bars = self.data.get_bars(
            benchmark, self.settings.risk_timeframe, self.settings.risk_lookback_days, as_of=as_of
        )
        closes = [] if bars.empty else bars["close"].tolist()
        price = closes[-1] if closes else None
        if price is None or price <= 0:
            return None

        hedge_dollars = gross_long * self.settings.hedge_ratio
        shares = int(hedge_dollars // price)
        if shares <= 0:
            return None
        return HedgeOrder(
            ticker=benchmark, side="short", shares=shares, dollars=shares * price
        )
```

- [ ] **Step 4: Wire the package export**

Replace the contents of `src/moneybot/risk/__init__.py` with:

```python
"""Risk Engine: deterministic sizing and hard-limit enforcement (no LLM)."""

from moneybot.risk.engine import RiskEngine

__all__ = ["RiskEngine"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/risk/test_engine.py -v`
Expected: PASS (all engine tests pass).

- [ ] **Step 6: Lint**

Run: `uv run ruff check src/moneybot/risk/`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/moneybot/risk/engine.py src/moneybot/risk/__init__.py tests/risk/test_engine.py
git commit -m "feat(risk): optional benchmark hedge + export RiskEngine"
```

---

## Task 9: Factory + README

**Files:**
- Create: `src/moneybot/risk/factory.py`
- Modify: `README.md`
- Test: `tests/risk/test_factory.py`

- [ ] **Step 1: Write the failing test**

Create `tests/risk/test_factory.py`:

```python
from moneybot.config import Settings, TickerMeta, Universe
from moneybot.risk.engine import RiskEngine
from moneybot.risk.factory import build_risk_engine


class _Prices:
    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        import pandas as pd
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])


def _datalayer(tmp_path):
    from moneybot.cache import Cache
    from moneybot.data_layer import DataLayer
    uni = Universe(sector="semiconductors", benchmark="SMH",
                   tickers=[TickerMeta(symbol="NVDA")])
    return DataLayer(uni, _Prices(), Cache(tmp_path))


def test_build_risk_engine_resolves_active_strategy(tmp_path):
    settings = Settings(strategy="catalyst_driven")
    engine = build_risk_engine(settings=settings, data_layer=_datalayer(tmp_path))
    assert isinstance(engine, RiskEngine)
    assert engine.strategy.name == "catalyst_driven"
    assert engine.settings is settings
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/risk/test_factory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.risk.factory'`.

- [ ] **Step 3: Write the factory**

Create `src/moneybot/risk/factory.py`:

```python
"""Wire a RiskEngine from settings: resolve the active strategy.

The Risk Engine uses no LLM, so unlike the agent factories there is no client to
construct — it just resolves the configured strategy (for its per-name/sector
caps) and hands over the data layer and settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import moneybot.strategies  # noqa: F401  -- import for side-effect: registers strategies
from moneybot.risk.engine import RiskEngine
from moneybot.strategies import registry

if TYPE_CHECKING:
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer


def build_risk_engine(
    *,
    settings: Settings,
    data_layer: DataLayer,
) -> RiskEngine:
    strategy = registry.get(settings.strategy)
    return RiskEngine(data_layer=data_layer, strategy=strategy, settings=settings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/risk/test_factory.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Add the README bullet**

In `README.md`, after the `Phase 6: analyst agent` bullet (the block ending `...that is the Risk Engine, Phase 7).`), add — matching the plain-prose style of Phases 1–6 (no bold lead-in):

```markdown
- Phase 7: risk engine — a deterministic, pure-Python layer (moneybot.risk) the agents
  cannot bypass. It takes the Analyst's TradePlans plus a portfolio snapshot and approves,
  downsizes, or vetoes each against hard limits: a kill switch and daily-loss circuit
  breaker halt all new entries; per name it blocks pyramiding and earnings-window entries,
  checks liquidity and price sanity, then sizes by conviction scaled down for volatility,
  bounded by per-name, sector-exposure, and cash caps. Every decision records the rule that
  fired, and an optional SMH hedge offsets sector beta when enabled. No LLM, no network.
```

- [ ] **Step 6: Lint**

Run: `uv run ruff check src/moneybot/risk/factory.py tests/risk/test_factory.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/moneybot/risk/factory.py tests/risk/test_factory.py README.md
git commit -m "feat(risk): add build_risk_engine factory + README phase 7"
```

---

## Final Verification

- [ ] **Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass (169 prior + the ~37 added here ≈ 206 passed), zero failures, no network access.

- [ ] **Lint the whole package**

Run: `uv run ruff check src/moneybot/risk/ tests/risk/ src/moneybot/config.py`
Expected: no errors.

- [ ] **Confirm the load-bearing invariants by inspection**

1. **Pure Python, no LLM:** `src/moneybot/risk/` imports no `moneybot.llm.*` and no `anthropic`. `grep -r "llm\|anthropic" src/moneybot/risk/` returns nothing.
2. **No network in tests:** every `tests/risk/` test uses fakes/`tmp_path`; none constructs a real provider or client.
3. **No fabricated clock:** `engine.py` contains no `date.today()`/`datetime.now()`; earnings blackout depends solely on the passed `as_of` (skipped when `None`).
4. **Point-in-time discipline:** every `get_bars` call in `engine.py` threads `as_of` through.
5. **Agents cannot bypass:** the engine sizes/vetoes purely from `StrategyParams` + `Settings` + portfolio + prices; Analyst conviction only *scales within* `max_position_pct`, never above it.
6. **Hard global halts:** kill switch and the daily-loss breaker short-circuit before any per-plan sizing and veto every plan.
7. **Every decision is auditable:** each `RiskDecision` carries `rules_fired`; downsizes append the binding rule, vetoes set exactly the rule that fired.

---

## Self-Review (run by the plan author before execution)

**1. Spec coverage (design §4.5, strategy §5–7):**
- Position sizing, volatility-scaled → Task 4 (`sizing.target_weight`) + Task 7 (wired).
- Max % per name → `StrategyParams.max_position_pct`, enforced in `target_weight` + Task 7 clamp.
- Sector/portfolio caps → Task 7 running `sector_exposure_cap` (single-sector bot ⇒ gross long = sector exposure).
- Daily loss circuit breaker → Task 6.
- Earnings blackout → Task 7 (`_in_earnings_blackout`, uses `TickerMeta.earnings_date`).
- Liquidity/sanity → Task 7 (min $-volume, price sanity, rounds-to-zero).
- Kill switch → Task 5 + Task 6.
- Decision logs the rule that fired → `RiskDecision.rules_fired` throughout.
- Conviction influences sizing within limits, never beyond → `target_weight` cap + Task 7.
- Stop-loss/profit-target/time-stop enforcement → these live on `ExitPlan` carried by each `TradePlan`; the Risk Engine *passes them through* approved decisions (it owns *entry* sizing/approval). Mechanical exit *execution* against live positions is the Execution/Orchestrator layer's job (Plans 8–9), which consumes `ExitPlan`. **No gap:** the spec assigns enforcement to the deterministic layer; the values are defined and carried, and the actual sell-trigger loop is out of scope for an entry-time sizing engine and explicitly belongs to the orchestrator that holds the position/price feed. Noted so the reviewer does not flag it as missing.
- Optional SMH hedge → Task 8.

**2. Placeholder scan:** No "TBD"/"handle edge cases"/uncoded steps — every code step shows complete code; the only English-described step is the README prose (intended).

**3. Type consistency:** `target_weight` keyword args match between Task 4 (definition) and Task 7 (call). `RiskDecision`/`PortfolioState`/`HedgeOrder` fields used in tests match Task 2 definitions. `params` is `StrategyParams` (`max_position_pct`, `max_sector_exposure_pct`, `hedge_enabled`) — names verified against `strategies/models.py`. `assess(plans, portfolio, as_of=None)` signature is identical across Tasks 6–8. `_veto`/`_assess_plan`/`_hedge`/`_in_earnings_blackout` names are consistent across tasks.

> **Note on incremental `engine.py`:** Tasks 6→7→8 build one file across three tasks (the import block and the tail of `assess` are edited each time). This mirrors how `analyst/agent.py` was built. Each task's "replace X with Y" anchors are exact strings from the prior task — apply them against the on-disk file.
