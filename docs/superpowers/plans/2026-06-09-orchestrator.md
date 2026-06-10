# Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Orchestrator — the conductor that runs one full trading cycle end-to-end (kill-switch / market-hours gate → mechanical exits → research → analyst → portfolio snapshot → risk engine → entry execution → journal → reconcile) and a factory that wires the whole bot from settings.

**Architecture:** A new `moneybot.orchestrator` package. The `Orchestrator` depends only on already-built collaborators (research agent, analyst agent, risk engine, execution adapter, journal, strategy, a start-of-day-equity store) plus an injected `clock` and `market_open` predicate — so every test injects fakes and never hits the network or an LLM. Two new pure-ish pieces carry the real logic: `build_portfolio_state` (translate the broker's positions + account into the Risk Engine's `PortfolioState`, marked to current prices, with today's P&L) and `evaluate_exits` (decide stop-loss / profit-target / time-stop on open longs). Exits run before entries and are placed through one new public `ExecutionAdapter.place` method. The trade journal is the source of truth for open-position entry dates.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, uv, ruff (line-length 100, self-checked — default rules don't flag E501). Conventions mirror the rest of moneybot: `from __future__ import annotations` in production modules (omitted in tests), type-only imports under `TYPE_CHECKING`, injected clock (no fabricated `datetime.now`/`date.today` inside components), `build_*(*, settings, ...)` factories with lazy LLM construction. NO test may hit the network or construct a real LLM client.

---

## Conventions every task must follow

- **Production modules** begin with `from __future__ import annotations`. **Test files do NOT.**
- **No fabricated clock.** The `Orchestrator` takes an injected `clock: Callable[[], datetime]`; tests pass a fixed clock. Components never call `datetime.now()`/`date.today()` directly.
- **Run tests:** `uv run pytest <path> -v`. **Lint:** `uv run ruff check src/moneybot/orchestrator tests/orchestrator`.
- **Sign convention** (matches `risk/models.py` and `execution/models.py`): long quantities positive, short negative.
- **Idempotency:** every order the orchestrator places carries a deterministic `client_order_id` derived from the cycle id, so a re-run within the same hour never double-trades (the adapter + store dedup).

---

## Interfaces this plan consumes (verified against the codebase — do not re-derive)

- `ResearchAgent.research_universe(as_of: date | None) -> dict[str, list[CatalystSignal]]`
- `AnalystAgent.analyze(research: dict, as_of: date | None) -> list[TradePlan]`
- `RiskEngine.assess(plans: list[TradePlan], portfolio: PortfolioState, as_of: date | None) -> RiskAssessment`
- `ExecutionAdapter.execute(assessment: RiskAssessment, cycle_id: str) -> list[Fill]`, `.reconcile() -> ReconciliationResult`, `.broker` (a `Broker`), `.store`
- `Broker.get_positions() -> list[PositionRecord]` (`ticker`, `qty` signed, `avg_price`), `.get_account() -> AccountSnapshot` (`equity`, `cash`)
- `JournalStore(root, clock=None)`, `.append(kind: str, ticker: str | None = None, payload: dict | None = None) -> JournalEntry`, `.read(ticker=None, kind=None, since=None) -> list[JournalEntry]` (entries have `.ts`, `.kind`, `.ticker`, `.payload`)
- `kill_switch_active(settings: Settings) -> bool` (from `moneybot.risk.kill_switch`)
- `Strategy.exit_plan() -> ExitPlan` (`max_hold_days: int`, `stop_loss_pct: float`, `profit_target_pct: float`, `thesis_check_guidance: str`)
- `TradePlan` carries `.ticker`, `.exit_plan: ExitPlan`
- `PortfolioState(equity: float [gt=0], cash: float, positions: list[Position], day_pnl_pct: float = 0.0)`; `Position(ticker: str, shares: float, market_value: float)`
- `OrderRequest(client_order_id, ticker, side, quantity: int [gt=0], order_type="market", reference_price: float | None [ge=0])`; `Fill(...)`, `ReconciliationResult(in_sync, discrepancies)`
- `Settings`: `data_dir`, `strategy`, `risk_timeframe`, `risk_lookback_days` (plus existing analyst/risk/execution fields)
- `DataLayer.get_bars(ticker, timeframe, lookback, as_of=None) -> DataFrame` (columns incl. `close`; RAISES `ValueError` for tickers not in `universe.symbols` and not the `benchmark`); `.universe.symbols`, `.universe.benchmark`, `.universe.sector`
- Factories: `build_research_agent(*, settings, data_layer, retriever, llm=None)`, `build_analyst_agent(*, settings, data_layer, retriever=None, llm=None)`, `build_risk_engine(*, settings, data_layer)`, `build_execution_adapter(*, settings, broker=None, store=None)`; `moneybot.strategies.registry.get(name)`

---

## File Structure

- Create: `src/moneybot/orchestrator/__init__.py` — package marker + public exports (exports added in Task 11)
- Create: `src/moneybot/orchestrator/models.py` — `ExitSignal`, `CycleResult`
- Create: `src/moneybot/orchestrator/market_hours.py` — `is_market_open`
- Create: `src/moneybot/orchestrator/portfolio.py` — `last_finite_close`, `mark_price`, `SodEquityStore`, `build_portfolio_state`
- Create: `src/moneybot/orchestrator/exits.py` — pure `evaluate_exits`
- Modify: `src/moneybot/execution/adapter.py` — add public `place(order) -> Fill`
- Create: `src/moneybot/orchestrator/engine.py` — the `Orchestrator` class
- Create: `src/moneybot/orchestrator/factory.py` — `build_orchestrator`
- Modify: `README.md` — Phase 9 bullet
- Test: `tests/orchestrator/__init__.py` + one test module per source file

---

### Task 1: Package + models

**Files:**
- Create: `src/moneybot/orchestrator/__init__.py`
- Create: `src/moneybot/orchestrator/models.py`
- Create: `tests/orchestrator/__init__.py` (empty)
- Test: `tests/orchestrator/test_models.py`

- [ ] **Step 1: Create the package markers**

Create `tests/orchestrator/__init__.py` empty. Create `src/moneybot/orchestrator/__init__.py`:

```python
"""Orchestrator: runs one full trading cycle end-to-end and wires the bot."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/orchestrator/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from moneybot.orchestrator.models import CycleResult, ExitSignal


def test_exit_signal_fields():
    s = ExitSignal(ticker="NVDA", shares=10, reason="stop_loss", reference_price=90.0)
    assert s.ticker == "NVDA" and s.shares == 10 and s.reason == "stop_loss"


def test_exit_signal_rejects_unknown_reason():
    with pytest.raises(ValidationError):
        ExitSignal(ticker="NVDA", shares=1, reason="vibes", reference_price=1.0)


def test_exit_signal_shares_must_be_positive():
    with pytest.raises(ValidationError):
        ExitSignal(ticker="NVDA", shares=0, reason="time_stop", reference_price=1.0)


def test_cycle_result_defaults():
    r = CycleResult(status="completed", cycle_id="2026-06-09T10")
    assert r.reason == ""
    assert r.entry_fills == [] and r.exit_fills == []
    assert r.plans_proposed == 0
    assert r.halted_by_risk is False
    assert r.reconciliation is None


def test_cycle_result_status_validated():
    with pytest.raises(ValidationError):
        CycleResult(status="exploded")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/orchestrator/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.orchestrator.models'`

- [ ] **Step 4: Implement the models**

Create `src/moneybot/orchestrator/models.py`:

```python
"""Result types for one orchestrator cycle.

ExitSignal is a triggered mechanical exit (stop/target/time-stop) the orchestrator
turns into a sell order. CycleResult is the structured summary one run_cycle call
returns — rich enough for a later observability layer to render, without this layer
owning any presentation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from moneybot.execution.models import Fill, ReconciliationResult


class ExitSignal(BaseModel):
    """A mechanical exit fired on an open long position."""

    ticker: str
    shares: int = Field(gt=0)  # whole shares to sell (the open quantity)
    reason: Literal["stop_loss", "profit_target", "time_stop"]
    reference_price: float


class CycleResult(BaseModel):
    """Structured outcome of one orchestrator cycle."""

    status: Literal["completed", "halted", "skipped"]
    reason: str = ""  # why halted/skipped (e.g. "kill_switch", "market_closed")
    cycle_id: str = ""
    plans_proposed: int = 0  # TradePlans the analyst produced
    entry_fills: list[Fill] = Field(default_factory=list)
    exit_fills: list[Fill] = Field(default_factory=list)
    halted_by_risk: bool = False  # a global risk gate (kill switch / circuit breaker) fired
    reconciliation: ReconciliationResult | None = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/orchestrator/test_models.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/orchestrator/__init__.py src/moneybot/orchestrator/models.py tests/orchestrator/__init__.py tests/orchestrator/test_models.py
git commit -m "feat(orchestrator): package + ExitSignal/CycleResult models"
```

---

### Task 2: Market-hours gate

**Files:**
- Create: `src/moneybot/orchestrator/market_hours.py`
- Test: `tests/orchestrator/test_market_hours.py`

- [ ] **Step 1: Write the failing test**

Create `tests/orchestrator/test_market_hours.py`:

```python
from datetime import datetime
from zoneinfo import ZoneInfo

from moneybot.orchestrator.market_hours import is_market_open

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def test_open_midday_weekday():
    # Wednesday 2026-06-10, 10:00 ET
    assert is_market_open(datetime(2026, 6, 10, 10, 0, tzinfo=ET)) is True


def test_closed_before_open():
    assert is_market_open(datetime(2026, 6, 10, 9, 0, tzinfo=ET)) is False


def test_open_at_930():
    assert is_market_open(datetime(2026, 6, 10, 9, 30, tzinfo=ET)) is True


def test_closed_after_4pm():
    assert is_market_open(datetime(2026, 6, 10, 16, 1, tzinfo=ET)) is False


def test_closed_on_saturday():
    # Saturday 2026-06-13, midday
    assert is_market_open(datetime(2026, 6, 13, 12, 0, tzinfo=ET)) is False


def test_naive_or_utc_is_converted():
    # 14:00 UTC on a weekday == 10:00 ET (summer, EDT) -> open
    assert is_market_open(datetime(2026, 6, 10, 14, 0, tzinfo=UTC)) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/orchestrator/test_market_hours.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `src/moneybot/orchestrator/market_hours.py`:

```python
"""A simple US-equity market-hours check.

Phase-1: regular session only — Mon-Fri, 09:30-16:00 America/New_York. It does
NOT know about market holidays or half-days; that (or a broker-clock check) can
replace this predicate later — the orchestrator takes it as an injectable, so
nothing else changes. Naive datetimes are assumed UTC.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_OPEN = time(9, 30)
_CLOSE = time(16, 0)


def is_market_open(now: datetime) -> bool:
    if now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo("UTC"))
    et = now.astimezone(_ET)
    if et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return _OPEN <= et.time() <= _CLOSE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/orchestrator/test_market_hours.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/orchestrator/market_hours.py tests/orchestrator/test_market_hours.py
git commit -m "feat(orchestrator): simple market-hours gate (injectable)"
```

---

### Task 3: Start-of-day equity store (day P&L)

**Files:**
- Create: `src/moneybot/orchestrator/portfolio.py` (this task adds `SodEquityStore`; Task 4 adds the rest to the same file)
- Test: `tests/orchestrator/test_sod_equity.py`

- [ ] **Step 1: Write the failing test**

Create `tests/orchestrator/test_sod_equity.py`:

```python
from datetime import date

from moneybot.orchestrator.portfolio import SodEquityStore


def test_first_call_of_day_returns_zero_pnl(tmp_path):
    store = SodEquityStore(tmp_path)
    # first observation of the day anchors start-of-day equity -> 0% P&L
    assert store.day_pnl_pct(100_000.0, date(2026, 6, 10)) == 0.0


def test_same_day_computes_pnl_against_anchor(tmp_path):
    store = SodEquityStore(tmp_path)
    store.day_pnl_pct(100_000.0, date(2026, 6, 10))  # anchor
    # equity fell to 97,000 -> -3%
    assert store.day_pnl_pct(97_000.0, date(2026, 6, 10)) == -0.03


def test_anchor_persists_across_instances(tmp_path):
    SodEquityStore(tmp_path).day_pnl_pct(100_000.0, date(2026, 6, 10))
    reopened = SodEquityStore(tmp_path)
    assert reopened.day_pnl_pct(110_000.0, date(2026, 6, 10)) == 0.1


def test_new_day_reanchors(tmp_path):
    store = SodEquityStore(tmp_path)
    store.day_pnl_pct(100_000.0, date(2026, 6, 10))
    # next day, equity is 90,000 -> that becomes the new anchor -> 0%
    assert store.day_pnl_pct(90_000.0, date(2026, 6, 11)) == 0.0
    assert store.day_pnl_pct(85_500.0, date(2026, 6, 11)) == -0.05


def test_zero_anchor_is_safe(tmp_path):
    store = SodEquityStore(tmp_path)
    # a degenerate 0 anchor must not divide by zero
    assert store.day_pnl_pct(0.0, date(2026, 6, 10)) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/orchestrator/test_sod_equity.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `src/moneybot/orchestrator/portfolio.py`:

```python
"""Build the Risk Engine's PortfolioState from the broker, marked to market.

SodEquityStore remembers the first equity seen each trading day so the Risk
Engine's daily-loss circuit breaker has a day_pnl_pct to read. build_portfolio_state
translates broker positions + account into a PortfolioState, marking each holding
to its current price via the data layer (falling back to cost when a price is
missing, and never asking the data layer about a non-universe ticker).
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from moneybot.risk.models import PortfolioState, Position

if TYPE_CHECKING:
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.execution.broker import Broker


class SodEquityStore:
    """Anchors start-of-day equity (JSON) to compute intraday P&L percentage."""

    def __init__(self, root: str | Path) -> None:
        self.path = Path(root) / "sod_equity.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def day_pnl_pct(self, equity: float, today: date) -> float:
        anchor = self._read(today)
        if anchor is None:
            self._write(today, equity)
            return 0.0
        if anchor <= 0:
            return 0.0
        return (equity - anchor) / anchor

    def _read(self, today: date) -> float | None:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("date") != today.isoformat():
            return None
        return float(data["equity"])

    def _write(self, today: date, equity: float) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"date": today.isoformat(), "equity": equity}), encoding="utf-8"
        )
        tmp.replace(self.path)


def last_finite_close(closes: list) -> float | None:
    return next((c for c in reversed(closes) if c is not None and math.isfinite(c)), None)


def mark_price(
    *,
    data_layer: DataLayer,
    ticker: str,
    timeframe: str,
    lookback: int,
    as_of: date | None,
) -> float | None:
    """Most recent finite close for a ticker, or None. Caller must ensure the
    ticker is in the universe (or is the benchmark) before calling."""
    bars = data_layer.get_bars(ticker, timeframe, lookback, as_of=as_of)
    closes = [] if bars.empty else bars["close"].tolist()
    return last_finite_close(closes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/orchestrator/test_sod_equity.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/orchestrator/portfolio.py tests/orchestrator/test_sod_equity.py
git commit -m "feat(orchestrator): start-of-day equity store + price-mark helpers"
```

---

### Task 4: build_portfolio_state

**Files:**
- Modify: `src/moneybot/orchestrator/portfolio.py` (append `build_portfolio_state`)
- Test: `tests/orchestrator/test_portfolio.py`

- [ ] **Step 1: Write the failing test**

Create `tests/orchestrator/test_portfolio.py`:

```python
import pandas as pd

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/orchestrator/test_portfolio.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_portfolio_state'`

- [ ] **Step 3: Implement (append to `portfolio.py`)**

Append to `src/moneybot/orchestrator/portfolio.py`:

```python


def build_portfolio_state(
    *,
    broker: Broker,
    data_layer: DataLayer,
    settings: Settings,
    as_of: date | None,
    day_pnl_pct: float,
) -> PortfolioState:
    """Translate the broker snapshot into the Risk Engine's PortfolioState.

    Each holding is marked to its current price (cost-basis fallback when a price
    is missing or the ticker is not markable). Equity comes from the broker; if the
    broker reports a non-positive equity, fall back to cash + marked value so the
    PortfolioState's gt=0 constraint holds.
    """
    account = broker.get_account()
    universe = data_layer.universe
    markable = set(universe.symbols) | {universe.benchmark}

    positions: list[Position] = []
    for rec in broker.get_positions():
        price: float | None = None
        if rec.ticker in markable:
            price = mark_price(
                data_layer=data_layer,
                ticker=rec.ticker,
                timeframe=settings.risk_timeframe,
                lookback=settings.risk_lookback_days,
                as_of=as_of,
            )
        if price is None:
            price = rec.avg_price  # mark-to-cost fallback
        positions.append(
            Position(ticker=rec.ticker, shares=rec.qty, market_value=rec.qty * price)
        )

    equity = account.equity
    if equity <= 0:
        equity = account.cash + sum(p.market_value for p in positions)

    return PortfolioState(
        equity=equity, cash=account.cash, positions=positions, day_pnl_pct=day_pnl_pct
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/orchestrator/test_portfolio.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/orchestrator/portfolio.py tests/orchestrator/test_portfolio.py
git commit -m "feat(orchestrator): build_portfolio_state (broker -> marked PortfolioState)"
```

---

### Task 5: evaluate_exits (pure)

**Files:**
- Create: `src/moneybot/orchestrator/exits.py`
- Test: `tests/orchestrator/test_exits.py`

- [ ] **Step 1: Write the failing test**

Create `tests/orchestrator/test_exits.py`:

```python
from datetime import date

from moneybot.execution.models import PositionRecord
from moneybot.orchestrator.exits import evaluate_exits
from moneybot.strategies.models import ExitPlan


def _plan(stop=0.08, target=0.20, max_hold=10):
    return ExitPlan(
        max_hold_days=max_hold,
        stop_loss_pct=stop,
        profit_target_pct=target,
        thesis_check_guidance="n/a",
    )


def _long(ticker="NVDA", qty=10.0, avg=100.0):
    return PositionRecord(ticker=ticker, qty=qty, avg_price=avg)


AS_OF = date(2026, 6, 20)
ENTRY = date(2026, 6, 18)  # 2 days before AS_OF


def test_no_trigger_in_band():
    sigs = evaluate_exits(
        positions=[_long()],
        entry_dates={"NVDA": ENTRY},
        current_prices={"NVDA": 105.0},  # +5%: below +20% target, above -8% stop
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert sigs == []


def test_stop_loss_triggers():
    sigs = evaluate_exits(
        positions=[_long()],
        entry_dates={"NVDA": ENTRY},
        current_prices={"NVDA": 92.0},  # -8% exactly -> stop
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert len(sigs) == 1
    assert sigs[0].reason == "stop_loss" and sigs[0].shares == 10
    assert sigs[0].reference_price == 92.0


def test_profit_target_triggers():
    sigs = evaluate_exits(
        positions=[_long()],
        entry_dates={"NVDA": ENTRY},
        current_prices={"NVDA": 120.0},  # +20% -> target
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert sigs[0].reason == "profit_target"


def test_time_stop_triggers_when_in_band_but_held_too_long():
    sigs = evaluate_exits(
        positions=[_long()],
        entry_dates={"NVDA": date(2026, 6, 1)},  # 19 days before AS_OF >= 10
        current_prices={"NVDA": 105.0},  # in band
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert sigs[0].reason == "time_stop"


def test_stop_loss_takes_precedence_over_time_stop():
    sigs = evaluate_exits(
        positions=[_long()],
        entry_dates={"NVDA": date(2026, 6, 1)},  # also past max hold
        current_prices={"NVDA": 80.0},  # also below stop
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert sigs[0].reason == "stop_loss"


def test_unknown_entry_date_skips_time_stop_only():
    # in band, no entry date -> cannot time-stop, so no signal
    sigs = evaluate_exits(
        positions=[_long()],
        entry_dates={},
        current_prices={"NVDA": 105.0},
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert sigs == []


def test_missing_price_skips_position():
    sigs = evaluate_exits(
        positions=[_long()],
        entry_dates={"NVDA": ENTRY},
        current_prices={},  # no mark available
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert sigs == []


def test_shorts_are_ignored():
    sigs = evaluate_exits(
        positions=[PositionRecord(ticker="SMH", qty=-5.0, avg_price=200.0)],
        entry_dates={"SMH": ENTRY},
        current_prices={"SMH": 100.0},
        exit_plan=_plan(),
        as_of=AS_OF,
    )
    assert sigs == []  # phase-1 exit loop manages longs only
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/orchestrator/test_exits.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `src/moneybot/orchestrator/exits.py`:

```python
"""Pure mechanical-exit evaluation for open long positions.

Given open positions, their entry dates, current prices, and the strategy's
ExitPlan, decide which longs to close and why. Precedence: stop-loss (capital
preservation) first, then profit-target, then time-stop. A position with no
current price is skipped (cannot evaluate); a position with no known entry date
can still stop/target but not time-stop. Phase-1 manages longs only — short
(hedge) lifecycle is deferred.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moneybot.orchestrator.models import ExitSignal

if TYPE_CHECKING:
    from datetime import date

    from moneybot.execution.models import PositionRecord
    from moneybot.strategies.models import ExitPlan


def evaluate_exits(
    *,
    positions: list[PositionRecord],
    entry_dates: dict[str, date],
    current_prices: dict[str, float],
    exit_plan: ExitPlan,
    as_of: date,
) -> list[ExitSignal]:
    signals: list[ExitSignal] = []
    for pos in positions:
        shares = int(pos.qty)
        if shares <= 0:  # longs only
            continue
        price = current_prices.get(pos.ticker)
        if price is None:
            continue

        entry = pos.avg_price
        reason: str | None = None
        if price <= entry * (1 - exit_plan.stop_loss_pct):
            reason = "stop_loss"
        elif price >= entry * (1 + exit_plan.profit_target_pct):
            reason = "profit_target"
        else:
            entry_date = entry_dates.get(pos.ticker)
            if entry_date is not None and (as_of - entry_date).days >= exit_plan.max_hold_days:
                reason = "time_stop"

        if reason is not None:
            signals.append(
                ExitSignal(
                    ticker=pos.ticker, shares=shares, reason=reason, reference_price=price
                )
            )
    return signals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/orchestrator/test_exits.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/orchestrator/exits.py tests/orchestrator/test_exits.py
git commit -m "feat(orchestrator): pure evaluate_exits (stop/target/time-stop, longs)"
```

---

### Task 6: Expose `ExecutionAdapter.place`

The orchestrator places exit sells through the adapter so the store stays consistent. The adapter already has a private `_place`; expose a public `place` (its docstring already anticipated exits flowing through it).

**Files:**
- Modify: `src/moneybot/execution/adapter.py`
- Test: `tests/execution/test_adapter.py` (add one test)

- [ ] **Step 1: Write the failing test (append to `tests/execution/test_adapter.py`)**

Append at the end of `tests/execution/test_adapter.py`:

```python
def test_place_sell_updates_store(tmp_path):
    from moneybot.execution.models import OrderRequest

    broker = FakeBroker()
    store = PositionStore(tmp_path)
    adapter = ExecutionAdapter(broker=broker, store=store)
    # open a long first
    adapter.execute(RiskAssessment(decisions=[_approved(shares=10, price=100.0)]), cycle_id="c1")
    # now place a direct sell of the whole position
    sell = OrderRequest(
        client_order_id="c2:NVDA:exit",
        ticker="NVDA",
        side="sell",
        quantity=10,
        reference_price=130.0,
    )
    fill = adapter.place(sell)
    assert fill.status == "filled" and fill.side == "sell"
    assert store.get_all() == []  # position closed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_adapter.py::test_place_sell_updates_store -v`
Expected: FAIL — `AttributeError: 'ExecutionAdapter' object has no attribute 'place'`

- [ ] **Step 3: Implement (modify `src/moneybot/execution/adapter.py`)**

Change the private `_place` into a public `place` and update its one caller. Replace:

```python
            fills.append(self._place(order))
```
(both occurrences, in `execute`) with:
```python
            fills.append(self.place(order))
```

And rename the method definition:

```python
    def _place(self, order: OrderRequest) -> Fill:
        fill = self.broker.place_order(order)
        if fill.status == "filled" and fill.filled_qty > 0:
            self.store.apply_fill(fill)
        return fill
```
to:
```python
    def place(self, order: OrderRequest) -> Fill:
        """Place a single order and update the store on a fill.

        Used by execute() for entries and by the orchestrator for exit sells.
        """
        fill = self.broker.place_order(order)
        if fill.status == "filled" and fill.filled_qty > 0:
            self.store.apply_fill(fill)
        return fill
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/execution/test_adapter.py -v`
Expected: PASS (all adapter tests, including the new one)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/execution/adapter.py tests/execution/test_adapter.py
git commit -m "feat(execution): expose ExecutionAdapter.place for direct (exit) orders"
```

---

### Task 7: Orchestrator — construction + gating

Build the `Orchestrator` incrementally. This task: constructor + the kill-switch and market-hours gates. `run_cycle` returns a halted/skipped `CycleResult` for those paths; the full pipeline arrives in Tasks 8-9.

**Files:**
- Create: `src/moneybot/orchestrator/engine.py`
- Test: `tests/orchestrator/test_engine_gating.py`

- [ ] **Step 1: Write the failing test**

Create `tests/orchestrator/test_engine_gating.py`:

```python
from datetime import datetime
from zoneinfo import ZoneInfo

from moneybot.orchestrator.engine import Orchestrator

ET = ZoneInfo("America/New_York")


class FakeJournal:
    def __init__(self):
        self.entries = []

    def append(self, kind, ticker=None, payload=None):
        self.entries.append((kind, ticker, payload or {}))
        return None

    def read(self, ticker=None, kind=None, since=None):
        return []


class _Settings:
    kill_switch_file = "this-file-does-not-exist"
    data_dir = "."
    risk_timeframe = "1d"
    risk_lookback_days = 20


def _orch(*, clock, market_open, journal=None, **kw):
    # Collaborators that must NOT be called on the gated paths get None;
    # if the gate is wrong, the test crashes on a None attribute access.
    return Orchestrator(
        settings=_Settings(),
        data_layer=kw.get("data_layer"),
        research=kw.get("research"),
        analyst=kw.get("analyst"),
        risk=kw.get("risk"),
        execution=kw.get("execution"),
        journal=journal or FakeJournal(),
        sod_equity=kw.get("sod_equity"),
        strategy=kw.get("strategy"),
        clock=clock,
        market_open=market_open,
    )


def test_kill_switch_halts(monkeypatch):
    monkeypatch.setenv("MONEYBOT_KILL_SWITCH", "1")
    journal = FakeJournal()
    orch = _orch(
        clock=lambda: datetime(2026, 6, 10, 10, 0, tzinfo=ET),
        market_open=lambda now: True,
        journal=journal,
    )
    result = orch.run_cycle()
    assert result.status == "halted" and result.reason == "kill_switch"
    assert ("halt", None, {"reason": "kill_switch"}) in journal.entries


def test_market_closed_skips():
    journal = FakeJournal()
    orch = _orch(
        clock=lambda: datetime(2026, 6, 13, 12, 0, tzinfo=ET),  # Saturday
        market_open=lambda now: False,
        journal=journal,
    )
    result = orch.run_cycle()
    assert result.status == "skipped" and result.reason == "market_closed"


def test_cycle_id_is_derived_from_clock():
    orch = _orch(
        clock=lambda: datetime(2026, 6, 10, 14, 0, tzinfo=ET),
        market_open=lambda now: False,  # skip early so no collaborators run
    )
    result = orch.run_cycle()
    assert result.cycle_id == "2026-06-10T14"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/orchestrator/test_engine_gating.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `src/moneybot/orchestrator/engine.py`:

```python
"""Orchestrator: run one full trading cycle end-to-end.

Order of operations each cycle: kill-switch gate -> market-hours gate -> mechanical
exits (close triggered longs) -> research -> analyst -> portfolio snapshot -> risk
engine -> entry execution -> reconcile, journaling each step. Every collaborator is
injected, so tests use fakes and nothing hits the network or an LLM. The clock is
injected (no fabricated time); the cycle_id is derived from it so a re-run within
the same hour is idempotent at the broker/store.

This task implements construction + the two global gates; exits and the entry
pipeline are added next.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from moneybot.orchestrator.models import CycleResult
from moneybot.risk.kill_switch import kill_switch_active

if TYPE_CHECKING:
    from datetime import datetime

    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.orchestrator.portfolio import SodEquityStore
    from moneybot.strategies.base import Strategy


class Orchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        data_layer: DataLayer,
        research,
        analyst,
        risk,
        execution,
        journal,
        sod_equity: SodEquityStore,
        strategy: Strategy,
        clock: Callable[[], datetime],
        market_open: Callable[[datetime], bool],
    ) -> None:
        self.settings = settings
        self.data = data_layer
        self.research = research
        self.analyst = analyst
        self.risk = risk
        self.execution = execution
        self.journal = journal
        self.sod_equity = sod_equity
        self.strategy = strategy
        self._clock = clock
        self._market_open = market_open

    def run_cycle(self, as_of=None) -> CycleResult:
        now = self._clock()
        cycle_id = now.strftime("%Y-%m-%dT%H")

        if kill_switch_active(self.settings):
            self.journal.append("halt", None, {"reason": "kill_switch"})
            return CycleResult(status="halted", reason="kill_switch", cycle_id=cycle_id)

        if not self._market_open(now):
            self.journal.append("skip", None, {"reason": "market_closed"})
            return CycleResult(status="skipped", reason="market_closed", cycle_id=cycle_id)

        # Exits + entry pipeline added in Tasks 8-9.
        return CycleResult(status="completed", cycle_id=cycle_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/orchestrator/test_engine_gating.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/orchestrator/engine.py tests/orchestrator/test_engine_gating.py
git commit -m "feat(orchestrator): Orchestrator construction + kill-switch/market-hours gates"
```

---

### Task 8: Orchestrator — exit phase

Add the exit phase: read open longs from the broker, mark prices, look up entry dates from the journal, evaluate exits, place sells, journal each.

**Files:**
- Modify: `src/moneybot/orchestrator/engine.py`
- Test: `tests/orchestrator/test_engine_exits.py`

- [ ] **Step 1: Write the failing test**

Create `tests/orchestrator/test_engine_exits.py`:

```python
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
    data = FakeData()
    data.price = 90.0  # NVDA bought at 100 -> -10% < -8% stop
    broker = FakeBroker([PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)])
    execution = FakeExecution(broker)
    journal = FakeJournal([_buy_entry("NVDA", datetime(2026, 6, 18, tzinfo=timezone.utc))])
    orch = _orch(data, execution, journal, FakeStrategy())

    fills = orch._run_exits(cycle_id="2026-06-20T10", as_of_date=__import__("datetime").date(2026, 6, 20))

    assert len(fills) == 1
    order = execution.placed[0]
    assert order.side == "sell" and order.ticker == "NVDA" and order.quantity == 10
    assert order.client_order_id == "2026-06-20T10:NVDA:exit"
    assert any(k == "exit" and t == "NVDA" for k, t, _ in journal.appended)


def test_run_exits_noop_when_in_band():
    data = FakeData()
    data.price = 103.0  # +3%, no trigger
    broker = FakeBroker([PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)])
    execution = FakeExecution(broker)
    journal = FakeJournal([_buy_entry("NVDA", datetime(2026, 6, 18, tzinfo=timezone.utc))])
    orch = _orch(data, execution, journal, FakeStrategy())

    import datetime as _dt
    fills = orch._run_exits(cycle_id="c", as_of_date=_dt.date(2026, 6, 20))
    assert fills == [] and execution.placed == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/orchestrator/test_engine_exits.py -v`
Expected: FAIL — `AttributeError: ... has no attribute '_run_exits'`

- [ ] **Step 3: Implement (modify `engine.py`)**

Add these imports to the top of `engine.py` (runtime imports — needed for calls):

```python
from moneybot.execution.models import OrderRequest
from moneybot.orchestrator.exits import evaluate_exits
from moneybot.orchestrator.portfolio import mark_price
```

And under `TYPE_CHECKING` add:
```python
    from datetime import date
```

Add these methods to the `Orchestrator` class:

```python
    def _markable(self, ticker: str) -> bool:
        u = self.data.universe
        return ticker in u.symbols or ticker == u.benchmark

    def _mark(self, ticker: str, as_of: date | None) -> float | None:
        if not self._markable(ticker):
            return None
        return mark_price(
            data_layer=self.data,
            ticker=ticker,
            timeframe=self.settings.risk_timeframe,
            lookback=self.settings.risk_lookback_days,
            as_of=as_of,
        )

    def _entry_dates(self, tickers: list[str]) -> dict[str, date]:
        """Most recent buy-fill date per ticker, from the journal (source of truth)."""
        dates: dict[str, date] = {}
        for ticker in tickers:
            buys = [
                e
                for e in self.journal.read(ticker=ticker, kind="fill")
                if e.payload.get("side") == "buy"
            ]
            if buys:
                dates[ticker] = max(e.ts for e in buys).date()
        return dates

    def _run_exits(self, *, cycle_id: str, as_of_date: date) -> list:
        longs = [p for p in self.execution.broker.get_positions() if p.qty > 0]
        if not longs:
            return []
        current_prices: dict[str, float] = {}
        for p in longs:
            price = self._mark(p.ticker, None)
            if price is not None:
                current_prices[p.ticker] = price
        entry_dates = self._entry_dates([p.ticker for p in longs])
        signals = evaluate_exits(
            positions=longs,
            entry_dates=entry_dates,
            current_prices=current_prices,
            exit_plan=self.strategy.exit_plan(),
            as_of=as_of_date,
        )
        fills = []
        for sig in signals:
            order = OrderRequest(
                client_order_id=f"{cycle_id}:{sig.ticker}:exit",
                ticker=sig.ticker,
                side="sell",
                quantity=sig.shares,
                reference_price=sig.reference_price,
            )
            fill = self.execution.place(order)
            self.journal.append(
                "exit",
                sig.ticker,
                {"reason": sig.reason, "shares": sig.shares, "status": fill.status},
            )
            fills.append(fill)
        return fills
```

Note: `_run_exits` takes `as_of_date` separately because the time-stop needs a concrete date even when the cycle's `as_of` is `None` (live); the caller passes `as_of or now.date()`. The `_mark` call passes `as_of=None` for live current prices (Task 9 threads the real `as_of` through).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/orchestrator/test_engine_exits.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/orchestrator/engine.py tests/orchestrator/test_engine_exits.py
git commit -m "feat(orchestrator): mechanical exit phase (evaluate + place sells + journal)"
```

---

### Task 9: Orchestrator — full entry pipeline

Wire `run_cycle` to run exits then the entry pipeline: research → analyst → portfolio snapshot (with day P&L) → risk → execute entries → journal buy fills (with exit plan) → reconcile.

**Files:**
- Modify: `src/moneybot/orchestrator/engine.py`
- Test: `tests/orchestrator/test_engine_cycle.py`

- [ ] **Step 1: Write the failing test**

Create `tests/orchestrator/test_engine_cycle.py`:

```python
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from moneybot.analyst.models import TradePlan
from moneybot.config import TickerMeta, Universe
from moneybot.execution.models import AccountSnapshot, Fill, ReconciliationResult
from moneybot.orchestrator.engine import Orchestrator
from moneybot.risk.models import RiskAssessment, RiskDecision
from moneybot.strategies.models import ExitPlan

ET = ZoneInfo("America/New_York")


class FakeData:
    def __init__(self):
        self.universe = Universe(
            sector="semis", benchmark="SMH",
            tickers=[TickerMeta(symbol="NVDA"), TickerMeta(symbol="AMD")],
        )

    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        return pd.DataFrame({"close": [100.0]})


class FakeBroker:
    def get_positions(self):
        return []  # no open positions -> exit phase is a no-op

    def get_account(self):
        return AccountSnapshot(equity=100_000.0, cash=100_000.0)


class FakeResearch:
    def __init__(self):
        self.called_with = None

    def research_universe(self, as_of=None):
        self.called_with = as_of
        return {"NVDA": []}


def _exit_plan():
    return ExitPlan(
        max_hold_days=10, stop_loss_pct=0.08, profit_target_pct=0.20,
        thesis_check_guidance="n/a",
    )


class FakeAnalyst:
    def analyze(self, research, as_of=None):
        return [
            TradePlan(
                ticker="NVDA", action="buy", conviction=0.7, thesis="t", score=1.0,
                exit_plan=_exit_plan(), analyst_note="ok",
            )
        ]


class FakeRisk:
    def __init__(self):
        self.portfolio = None

    def assess(self, plans, portfolio, as_of=None):
        self.portfolio = portfolio
        return RiskAssessment(
            decisions=[
                RiskDecision(
                    ticker="NVDA", approved=True, shares=10, target_dollars=1000.0,
                    target_weight=0.01, reference_price=100.0, reasoning="ok",
                )
            ]
        )


class FakeExecution:
    def __init__(self, broker):
        self.broker = broker
        self.executed = None

    def place(self, order):
        raise AssertionError("no exits expected in this test")

    def execute(self, assessment, cycle_id):
        self.executed = (assessment, cycle_id)
        return [
            Fill(
                client_order_id=f"{cycle_id}:NVDA:buy", broker_order_id="b1",
                ticker="NVDA", side="buy", status="filled", filled_qty=10,
                avg_price=100.0, ts=datetime(2026, 6, 10, tzinfo=timezone.utc),
            )
        ]

    def reconcile(self):
        return ReconciliationResult(in_sync=True, discrepancies=[])


class FakeJournal:
    def __init__(self):
        self.appended = []

    def append(self, kind, ticker=None, payload=None):
        self.appended.append((kind, ticker, payload or {}))
        return None

    def read(self, ticker=None, kind=None, since=None):
        return []


class FakeSod:
    def day_pnl_pct(self, equity, today):
        return -0.01


class FakeStrategy:
    def exit_plan(self):
        return _exit_plan()


class _Settings:
    kill_switch_file = "this-file-does-not-exist"
    risk_timeframe = "1d"
    risk_lookback_days = 20


def _orch():
    research = FakeResearch()
    risk = FakeRisk()
    execution = FakeExecution(FakeBroker())
    journal = FakeJournal()
    orch = Orchestrator(
        settings=_Settings(), data_layer=FakeData(), research=research,
        analyst=FakeAnalyst(), risk=risk, execution=execution, journal=journal,
        sod_equity=FakeSod(), strategy=FakeStrategy(),
        clock=lambda: datetime(2026, 6, 10, 10, 0, tzinfo=ET),
        market_open=lambda now: True,
    )
    return orch, research, risk, execution, journal


def test_full_cycle_runs_pipeline_and_executes_entries():
    orch, research, risk, execution, journal = _orch()
    result = orch.run_cycle()

    assert result.status == "completed" and result.cycle_id == "2026-06-10T10"
    assert result.plans_proposed == 1
    assert len(result.entry_fills) == 1 and result.entry_fills[0].ticker == "NVDA"
    assert result.reconciliation is not None and result.reconciliation.in_sync
    # risk engine saw a portfolio with the day P&L from the SOD store
    assert risk.portfolio.day_pnl_pct == -0.01
    # the execution adapter was handed the assessment + cycle id
    assert execution.executed[1] == "2026-06-10T10"
    # a buy fill was journaled with the exit plan for later time-stop lookup
    buy_journals = [p for k, t, p in journal.appended if k == "fill" and t == "NVDA"]
    assert buy_journals and buy_journals[0]["side"] == "buy"
    assert "exit_plan" in buy_journals[0]


def test_halted_assessment_sets_flag():
    research = FakeResearch()
    execution = FakeExecution(FakeBroker())

    class HaltRisk:
        def assess(self, plans, portfolio, as_of=None):
            return RiskAssessment(decisions=[], halted=True)

    orch = Orchestrator(
        settings=_Settings(), data_layer=FakeData(), research=research,
        analyst=FakeAnalyst(), risk=HaltRisk(), execution=execution,
        journal=FakeJournal(), sod_equity=FakeSod(), strategy=FakeStrategy(),
        clock=lambda: datetime(2026, 6, 10, 10, 0, tzinfo=ET),
        market_open=lambda now: True,
    )
    result = orch.run_cycle()
    assert result.status == "completed" and result.halted_by_risk is True
    assert result.entry_fills == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/orchestrator/test_engine_cycle.py -v`
Expected: FAIL — assertion errors (run_cycle still returns the Task-7 stub `CycleResult` with no fills/pipeline)

- [ ] **Step 3: Implement (modify `engine.py`)**

Add to the runtime imports in `engine.py`:
```python
from moneybot.orchestrator.portfolio import build_portfolio_state, mark_price
```
(replace the existing `from moneybot.orchestrator.portfolio import mark_price` line with the combined import above).

Replace the gated `run_cycle` body's final line (`return CycleResult(status="completed", cycle_id=cycle_id)`) with the full pipeline:

```python
        as_of_date = as_of if as_of is not None else now.date()

        # 1. Mechanical exits on existing positions (before taking on new ones).
        exit_fills = self._run_exits(cycle_id=cycle_id, as_of_date=as_of_date)

        # 2. Research -> Analyst.
        research = self.research.research_universe(as_of=as_of)
        plans = self.analyst.analyze(research, as_of=as_of)
        self.journal.append("plans", None, {"count": len(plans)})

        # 3. Portfolio snapshot (marked to market) with today's P&L for the breaker.
        account = self.execution.broker.get_account()
        day_pnl = self.sod_equity.day_pnl_pct(account.equity, as_of_date)
        portfolio = build_portfolio_state(
            broker=self.execution.broker,
            data_layer=self.data,
            settings=self.settings,
            as_of=as_of,
            day_pnl_pct=day_pnl,
        )

        # 4. Risk Engine -> entry execution.
        assessment = self.risk.assess(plans, portfolio, as_of=as_of)
        entry_fills = self.execution.execute(assessment, cycle_id=cycle_id)

        # Journal buy fills with their exit plan so the exit loop can time-stop later.
        plan_by_ticker = {p.ticker: p for p in plans}
        for fill in entry_fills:
            if fill.status == "filled" and fill.side == "buy":
                plan = plan_by_ticker.get(fill.ticker)
                self.journal.append(
                    "fill",
                    fill.ticker,
                    {
                        "side": "buy",
                        "shares": fill.filled_qty,
                        "price": fill.avg_price,
                        "exit_plan": plan.exit_plan.model_dump() if plan else None,
                    },
                )

        reconciliation = self.execution.reconcile()
        return CycleResult(
            status="completed",
            cycle_id=cycle_id,
            plans_proposed=len(plans),
            entry_fills=entry_fills,
            exit_fills=exit_fills,
            halted_by_risk=assessment.halted,
            reconciliation=reconciliation,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/orchestrator/test_engine_cycle.py -v`
Expected: PASS (2 passed). Also re-run the gating + exits tests to confirm no regression: `uv run pytest tests/orchestrator -v`.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/orchestrator/engine.py tests/orchestrator/test_engine_cycle.py
git commit -m "feat(orchestrator): full run_cycle pipeline (exits -> research -> analyst -> risk -> execute)"
```

---

### Task 10: Factory

Wire the whole bot from settings. The caller supplies the `data_layer` and `retriever` (their construction involves providers/cache/universe and is out of this plan's scope); the factory builds the agents, risk engine, execution adapter, journal, SOD store, and orchestrator, threading a shared clock.

**Files:**
- Create: `src/moneybot/orchestrator/factory.py`
- Test: `tests/orchestrator/test_factory.py`

- [ ] **Step 1: Write the failing test**

Create `tests/orchestrator/test_factory.py`:

```python
from datetime import datetime, timezone

import pandas as pd

from moneybot.config import Settings, TickerMeta, Universe
from moneybot.memory.models import MemoryContext
from moneybot.orchestrator.engine import Orchestrator
from moneybot.orchestrator.factory import build_orchestrator


class FakeData:
    def __init__(self):
        self.universe = Universe(
            sector="semis", benchmark="SMH",
            tickers=[TickerMeta(symbol="NVDA")],
        )

    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        return pd.DataFrame({"close": [100.0]})


class FakeRetriever:
    def retrieve(self, tickers, sector):
        return MemoryContext()


class FakeLLM:
    def complete_json(self, *, model, system, user, schema):
        return {}


def test_build_orchestrator_wires_everything(tmp_path):
    settings = Settings(mode="paper", data_dir=str(tmp_path))
    orch = build_orchestrator(
        settings=settings,
        data_layer=FakeData(),
        retriever=FakeRetriever(),
        llm=FakeLLM(),
        clock=lambda: datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc),
    )
    assert isinstance(orch, Orchestrator)
    # collaborators are wired
    assert orch.research is not None and orch.analyst is not None
    assert orch.risk is not None and orch.execution is not None
    assert orch.journal is not None and orch.sod_equity is not None
    assert orch.strategy is not None
    # journal + sod store live under data_dir
    assert orch.journal.path == tmp_path / "journal.jsonl"
    assert orch.sod_equity.path == tmp_path / "sod_equity.json"


def test_market_open_defaults_to_real_predicate(tmp_path):
    settings = Settings(mode="paper", data_dir=str(tmp_path))
    orch = build_orchestrator(
        settings=settings, data_layer=FakeData(), retriever=FakeRetriever(),
        llm=FakeLLM(),
    )
    # Saturday -> closed via the default is_market_open
    from datetime import datetime as dt
    from zoneinfo import ZoneInfo
    assert orch._market_open(dt(2026, 6, 13, 12, 0, tzinfo=ZoneInfo("America/New_York"))) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/orchestrator/test_factory.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `src/moneybot/orchestrator/factory.py`:

```python
"""Wire the whole bot into an Orchestrator from settings.

The caller supplies the data layer and memory retriever (their construction —
providers, cache, universe, memory stores — is outside this plan). Everything
else is built here from the existing component factories, sharing one injected
clock so the journal's timestamps line up with the cycle (and a later backtest
can replay dated cycles). The LLM is optional: omit it in production to lazily
construct the real client, inject a fake in tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from moneybot.analyst.factory import build_analyst_agent
from moneybot.execution.factory import build_execution_adapter
from moneybot.memory.journal import JournalStore
from moneybot.orchestrator.engine import Orchestrator
from moneybot.orchestrator.market_hours import is_market_open
from moneybot.orchestrator.portfolio import SodEquityStore
from moneybot.research.factory import build_research_agent
from moneybot.risk.factory import build_risk_engine
from moneybot.strategies import registry

if TYPE_CHECKING:
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.llm.client import LLMClient
    from moneybot.memory.retriever import MemoryRetriever


def build_orchestrator(
    *,
    settings: Settings,
    data_layer: DataLayer,
    retriever: MemoryRetriever,
    llm: LLMClient | None = None,
    clock: Callable[[], datetime] | None = None,
    market_open: Callable[[datetime], bool] = is_market_open,
) -> Orchestrator:
    clock = clock or (lambda: datetime.now(timezone.utc))

    research = build_research_agent(
        settings=settings, data_layer=data_layer, retriever=retriever, llm=llm
    )
    analyst = build_analyst_agent(
        settings=settings, data_layer=data_layer, retriever=retriever, llm=llm
    )
    risk = build_risk_engine(settings=settings, data_layer=data_layer)
    execution = build_execution_adapter(settings=settings)
    journal = JournalStore(settings.data_dir, clock=clock)
    sod_equity = SodEquityStore(settings.data_dir)
    strategy = registry.get(settings.strategy)

    return Orchestrator(
        settings=settings,
        data_layer=data_layer,
        research=research,
        analyst=analyst,
        risk=risk,
        execution=execution,
        journal=journal,
        sod_equity=sod_equity,
        strategy=strategy,
        clock=clock,
        market_open=market_open,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/orchestrator/test_factory.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/orchestrator/factory.py tests/orchestrator/test_factory.py
git commit -m "feat(orchestrator): build_orchestrator factory (wires the whole bot)"
```

---

### Task 11: Public exports + README

**Files:**
- Modify: `src/moneybot/orchestrator/__init__.py`
- Modify: `README.md`
- Test: `tests/orchestrator/test_exports.py`

- [ ] **Step 1: Write the failing test**

Create `tests/orchestrator/test_exports.py`:

```python
import moneybot.orchestrator as orch


def test_public_exports():
    assert orch.Orchestrator is not None
    assert orch.build_orchestrator is not None
    assert set(["Orchestrator", "build_orchestrator"]).issubset(orch.__all__)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/orchestrator/test_exports.py -v`
Expected: FAIL — `AttributeError: module 'moneybot.orchestrator' has no attribute 'Orchestrator'`

- [ ] **Step 3: Update the package init**

Replace the contents of `src/moneybot/orchestrator/__init__.py` with:

```python
"""Orchestrator: runs one full trading cycle end-to-end and wires the bot."""

from moneybot.orchestrator.engine import Orchestrator
from moneybot.orchestrator.factory import build_orchestrator

__all__ = ["Orchestrator", "build_orchestrator"]
```

- [ ] **Step 4: Add the README bullet**

In `README.md`, immediately after the Phase 8 (execution adapter) bullet, add:

```markdown
- **Phase 9: orchestrator** — the conductor (moneybot.orchestrator) that runs one full
  trading cycle end-to-end: it checks the kill switch and market hours, closes any
  positions that hit their stop-loss / profit-target / time-stop, then runs research →
  analyst → a marked-to-market portfolio snapshot → risk engine → entry execution, and
  journals every step before reconciling against the broker. Every component is injected,
  so the whole cycle runs in tests with fakes — no network, no LLM, an injected clock.
  `build_orchestrator` wires the entire bot from settings.
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/orchestrator/test_exports.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/orchestrator/__init__.py README.md tests/orchestrator/test_exports.py
git commit -m "feat(orchestrator): public exports + README phase 9"
```

---

### Task 12: Final verification + whole-implementation review + merge

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `uv run pytest -q`
Expected: PASS — all prior 277 tests plus the new orchestrator tests (models 5, market_hours 6, sod 5, portfolio 4, exits 8, engine_gating 3, engine_exits 2, engine_cycle 2, factory 2, exports 1 = ~38 new, plus the 1 added execution test) with no failures and no network access.

- [ ] **Step 2: Lint**

Run: `uv run ruff check src/moneybot tests`
Expected: no errors. Eyeball any long lines (ruff's default rule set does not flag E501).

- [ ] **Step 3: Verify the discipline invariants**

- No network / no LLM in orchestrator tests: confirm no test under `tests/orchestrator/` imports `anthropic`, `alpaca`, `requests`, `httpx`, or constructs a real `AnthropicClient` (all collaborators are fakes or injected).
- No fabricated clock: confirm `engine.py` uses only the injected `self._clock`; `market_hours.py`, `portfolio.py`, `exits.py` take time/dates as parameters; the factory's default clock is the only `datetime.now`.
- Run: `uv run python -c "import moneybot.orchestrator; print('import ok')"` — confirms the package imports cleanly.

- [ ] **Step 4: Dispatch the final whole-implementation review**

Review the entire `src/moneybot/orchestrator/` package + the `ExecutionAdapter.place` change against these invariants: (1) one cycle wires gate→exits→research→analyst→portfolio→risk→execute→journal→reconcile in the right order; (2) no network/LLM in tests; (3) no fabricated clock (injected everywhere); (4) idempotent orders (deterministic cycle_id → client_order_id); (5) exits are longs-only, stop>target>time precedence, skip on missing price/entry-date; (6) PortfolioState marks correctly (sign convention, unmarkable/zero-equity fallbacks), day P&L from the SOD store; (7) the journal is the source of truth for entry dates; (8) integration shapes match the real component method signatures. Fix anything the review surfaces (TDD: failing test → fix → green), commit, then re-verify the full suite.

- [ ] **Step 5: Merge** (use superpowers:finishing-a-development-branch)

Verify tests pass, then merge `plan-9/orchestrator` to `main` with `--no-ff` and a detailed message ending in the `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer; re-run the suite on merged `main`; push; delete the branch.

---

## Final Self-Review (run by the controller after writing, before execution)

**1. Spec coverage (§4 components + §5 data flow steps 1, 5b, 6, 8):**
- §5.1 "checks kill switch and market hours; starts cycle, opens journal entry" → Task 7 gates + journaling. ✓
- §5.6 "Execution Adapter: place approved orders; record fills" → Task 9 `execute` + journal buy fills. ✓
- "owns the mechanical exit trigger loop on open positions" (carried from Plan 7/8 notes) → Tasks 5, 8 (`evaluate_exits` + `_run_exits`). ✓
- "Orchestrator wiring + run journal" (build-sequence step 8) → Tasks 7-10; observability/alerting explicitly deferred (operator's choice). ✓
- PortfolioState construction (needed for §5.5 risk) → Tasks 3-4. ✓

**2. Cross-cutting invariants:**
- No network/LLM in tests — all collaborators injected as fakes; factory test injects a fake LLM. ✓
- No fabricated clock — `Orchestrator` uses injected `self._clock`; pure modules take dates as params; only the factory default is `datetime.now`. ✓
- Idempotency — `cycle_id = now.strftime("%Y-%m-%dT%H")`; exit orders keyed `"{cycle_id}:{ticker}:exit"`, entries keyed by the adapter. ✓
- Sign convention (long +, short -) — `build_portfolio_state` market_value = qty*price; `evaluate_exits` longs-only. ✓
- Journal as source of truth for entry dates — `_entry_dates` reads buy fills. ✓
- Data-layer ValueError on non-universe tickers handled — `_markable` guard before `mark_price`; cost fallback. ✓

**3. Type/name consistency check:**
- `Orchestrator.__init__` kwargs match the factory's construction and all three engine test files. ✓
- `build_portfolio_state(*, broker, data_layer, settings, as_of, day_pnl_pct)` — matches Task 4 test and the Task 9 call. ✓
- `evaluate_exits(*, positions, entry_dates, current_prices, exit_plan, as_of)` — matches Task 5 test and the `_run_exits` call. ✓
- `SodEquityStore.day_pnl_pct(equity, today)` — matches Task 3 test and Task 9 call. ✓
- `ExecutionAdapter.place(order)` — matches Task 6 test and `_run_exits`. ✓
- Consumes `RiskAssessment.halted/.decisions`, `Fill.status/.side/.ticker/.filled_qty/.avg_price`, `TradePlan.exit_plan`, `ExitPlan.stop_loss_pct/.profit_target_pct/.max_hold_days`, `PortfolioState`/`Position`, `JournalEntry.ts/.kind/.ticker/.payload` — all verified against the codebase. ✓
- `CycleResult`/`ExitSignal` fields consistent across `models.py`, `engine.py`, and tests. ✓

No gaps found.
